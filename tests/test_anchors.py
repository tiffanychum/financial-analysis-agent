"""Tests for anchors.py's own-history percentiles. data.py's SEC XBRL
fetch/parse machinery (fetch_sec_companyfacts, sec_ttm_history) is already
covered in test_data.py; here we mock at that boundary plus anchors' own
_price_series/_shares_series seam, so these tests exercise only the
percentile/multiple-series math this module owns.
"""
from __future__ import annotations

import pandas as pd
import pytest

import anchors as A
import data as D
import valuation as V

# 8 quarter-end dates, ~2 years - the MIN_QUARTERS floor.
_DATES = ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
         "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31"]


def _series(values: list[float]) -> pd.Series:
    return pd.Series({pd.Timestamp(d): v for d, v in zip(_DATES, values, strict=True)}).sort_index()


def test_percentile_rank_basic():
    assert A.percentile_rank(6.5, list(range(1, 9))) == pytest.approx(75.0)  # 6 of 8 <= 6.5


def test_percentile_rank_none_below_min_quarters():
    assert A.percentile_rank(50, [10, 20, 30]) is None  # fewer than MIN_QUARTERS


def test_percentile_rank_none_when_current_is_none():
    assert A.percentile_rank(None, list(range(20))) is None


def test_ps_percentile_computes_from_real_revenue_and_price_history(monkeypatch):
    # price_i = i+1 (1..8), shares=1, revenue=1 -> ps_i = i+1 exactly.
    revenue_history = [(d, 1.0) for d in _DATES]
    monkeypatch.setattr(D, "fetch_sec_companyfacts", lambda ticker: {"facts": {}})
    monkeypatch.setattr(D, "sec_ttm_history", lambda facts, tags: revenue_history)
    monkeypatch.setattr(A, "_price_series", lambda ticker, years_back=10: _series(list(range(1, 9))))
    monkeypatch.setattr(A, "_shares_series", lambda facts: _series([1.0] * 8))

    result = A.ps_percentile("TEST", {"market_cap": 6.5, "revenue": 1.0})
    assert result["applicable"] is True
    assert result["value"] == pytest.approx(6.5)
    assert result["percentile"] == pytest.approx(75.0)
    assert result["n"] == 8
    assert "SEC XBRL" in result["source"]


def test_ps_percentile_not_applicable_without_us_cik(monkeypatch):
    monkeypatch.setattr(D, "fetch_sec_companyfacts", lambda ticker: None)
    result = A.ps_percentile("IGLN.L", {"market_cap": 100.0, "revenue": 10.0})
    assert result == {"applicable": False, "reason": "no resolvable US SEC CIK"}


def test_ps_percentile_not_applicable_missing_current_fields(monkeypatch):
    monkeypatch.setattr(D, "fetch_sec_companyfacts", lambda ticker: {"facts": {}})
    monkeypatch.setattr(D, "sec_ttm_history", lambda facts, tags: [(d, 1.0) for d in _DATES])
    result = A.ps_percentile("TEST", {"market_cap": None, "revenue": 10.0})
    assert result["applicable"] is False and "market cap" in result["reason"]


def test_ps_percentile_too_few_quarters(monkeypatch):
    one_date = pd.Series({pd.Timestamp(_DATES[0]): 1.0})
    monkeypatch.setattr(D, "fetch_sec_companyfacts", lambda ticker: {"facts": {}})
    monkeypatch.setattr(D, "sec_ttm_history", lambda facts, tags: [(_DATES[0], 1.0)])  # only 1
    monkeypatch.setattr(A, "_price_series", lambda ticker, years_back=10: one_date)
    monkeypatch.setattr(A, "_shares_series", lambda facts: one_date)
    result = A.ps_percentile("TEST", {"market_cap": 6.5, "revenue": 1.0})
    assert result["applicable"] is False
    assert "need >= 8" in result["reason"]


def test_ev_fcf_percentile_combines_cfo_and_capex_and_holds_net_debt_fixed(monkeypatch):
    # cfo_i = i+2, capex_i = 1 (constant) -> fcf_i = i+1, matching _DATES i+1.
    cfo_history = [(d, i + 2.0) for i, d in enumerate(_DATES)]
    capex_history = [(d, 1.0) for d in _DATES]

    def fake_ttm_history(facts, tags):
        return cfo_history if tags == A._CFO_TAGS else capex_history

    monkeypatch.setattr(D, "fetch_sec_companyfacts", lambda ticker: {"facts": {}})
    monkeypatch.setattr(D, "sec_ttm_history", fake_ttm_history)
    monkeypatch.setattr(A, "_price_series", lambda ticker, years_back=10: _series(list(range(1, 9))))
    monkeypatch.setattr(A, "_shares_series", lambda facts: _series([1.0] * 8))

    # today: market_cap=6, net_debt=4 -> ev=10; fcf=1 -> ev/fcf=10, which
    # should land at the top of the fcf-1..8-implied ev/fcf history since
    # net_debt is held fixed at 4 for every historical point too.
    f = {"market_cap": 6.0, "total_debt": 4.0, "cash": 0.0, "fcf": 1.0}
    result = A.ev_fcf_percentile("TEST", f)
    assert result["applicable"] is True
    assert result["value"] == pytest.approx((6.0 + 4.0) / 1.0)
    assert result["n"] == 8


def test_ev_fcf_percentile_not_applicable_when_fcf_non_positive(monkeypatch):
    monkeypatch.setattr(D, "fetch_sec_companyfacts", lambda ticker: {"facts": {}})
    result = A.ev_fcf_percentile("TEST", {"market_cap": 100.0, "fcf": -5.0})
    assert result["applicable"] is False and "FCF" in result["reason"]


def test_growth_percentile_ranks_current_against_same_solver_historically(monkeypatch):
    # flat FCF=1.0/quarter (cfo=2 constant, capex=1 constant); rising EV
    # (price 8..22, net debt 0) -> a monotonically rising implied-growth
    # history, since solve_implied_growth(base, r, ev) is increasing in ev
    # for fixed base/r. Ordinary 8x-22x FCF multiples so every quarter
    # solves within the solver's [-50%, +100%]/yr range.
    cfo_history = [(d, 2.0) for d in _DATES]
    capex_history = [(d, 1.0) for d in _DATES]
    prices = [8.0 + 2.0 * i for i in range(8)]  # 8,10,...,22

    def fake_ttm_history(facts, tags):
        return cfo_history if tags == A._CFO_TAGS else capex_history

    monkeypatch.setattr(D, "fetch_sec_companyfacts", lambda ticker: {"facts": {}})
    monkeypatch.setattr(D, "sec_ttm_history", fake_ttm_history)
    monkeypatch.setattr(A, "_price_series", lambda ticker, years_back=10: _series(prices))
    monkeypatch.setattr(A, "_shares_series", lambda facts: _series([1.0] * 8))

    f = {"market_cap": prices[-1], "total_debt": 0.0, "cash": 0.0, "beta": 1.0}
    rf, erp = 0.04, 0.05
    r = V.wacc(f, rf, erp)
    # oracle: today's point is identical to the priciest historical quarter
    # (same fcf=1.0, same ev=prices[-1]) -> current_growth must land exactly
    # at the top of hist.
    current_growth, _resid, reason = V.solve_implied_growth(1.0, r, prices[-1])
    assert reason is None, "test fixture's ev/fcf multiple must be solver-reachable"

    result = A.growth_percentile("TEST", f, rf=rf, erp=erp, current_growth=current_growth)
    assert result["applicable"] is True
    assert result["value"] == pytest.approx(current_growth)
    assert result["percentile"] == pytest.approx(100.0)
    assert result["n"] == 8
    assert "today's WACC held fixed" in result["source"]


def test_growth_percentile_not_applicable_without_fcf_history(monkeypatch):
    monkeypatch.setattr(D, "fetch_sec_companyfacts", lambda ticker: {"facts": {}})
    monkeypatch.setattr(D, "sec_ttm_history", lambda facts, tags: [])
    result = A.growth_percentile("TEST", {}, rf=0.04, erp=0.05, current_growth=0.03)
    assert result["applicable"] is False
    assert "no historical FCF" in result["reason"]


def test_etf_price_percentile_uses_own_trailing_range(monkeypatch):
    n = 200
    prices = _series_n(n)
    monkeypatch.setattr(A, "_price_series", lambda ticker, years_back=10: prices)
    result = A.etf_price_percentile("QQQ")
    assert result["applicable"] is True
    assert result["percentile"] == pytest.approx(100.0)  # last value is the max
    assert "forward-P/E percentile proxy" in result["source"]


def test_etf_price_percentile_not_applicable_with_short_history(monkeypatch):
    monkeypatch.setattr(A, "_price_series", lambda ticker, years_back=10: _series_n(10))
    result = A.etf_price_percentile("NEWFUND")
    assert result == {"applicable": False, "reason": "insufficient price history"}


def _series_n(n: int) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(range(1, n + 1), index=idx, dtype=float)


def test_growth_percentile_not_applicable_without_share_history(monkeypatch):
    # Multi-class issuers (Alphabet) have no cover-page share count: must not crash.
    monkeypatch.setattr(D, "fetch_sec_companyfacts", lambda ticker: {"facts": {}})
    monkeypatch.setattr(A, "_fcf_ttm_history", lambda facts: [(d, 1.0) for d in _DATES])
    monkeypatch.setattr(A, "_price_series", lambda ticker, years_back=10: _series(list(range(1, 9))))
    monkeypatch.setattr(A, "_shares_series", lambda facts: pd.Series(dtype=float))
    out = A.growth_percentile("GOOG", {"total_debt": 0.0, "cash": 0.0}, 0.04, 0.05, 0.1)
    assert out["applicable"] is False and "share-count" in out["reason"]

"""Tests for portfolio.py: an entirely optional feature (no portfolio.yaml
-> no context, not an error), so load_portfolio's file-presence branch and
snapshot's valuation/weight/HHI math are the behaviors worth covering.
"""
from __future__ import annotations

import pytest

import data as D
import portfolio as P


def test_load_portfolio_returns_none_when_file_missing(tmp_path):
    assert P.load_portfolio(tmp_path / "nope.yaml") is None


def test_load_portfolio_parses_shares_and_value_holdings(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "currency: USD\ncash: 1000\nholdings:\n  AAPL:\n    shares: 10\n"
        "  EQQQ.L:\n    value: 500\ntargets:\n  AAPL: 20\n  cash: 10\n")
    out = P.load_portfolio(path)
    assert out == {"currency": "USD", "cash": 1000.0,
                   "holdings": {"AAPL": {"shares": 10.0, "value": None},
                               "EQQQ.L": {"shares": None, "value": 500.0}},
                   "targets": {"AAPL": 20.0, "cash": 10.0}}


def test_load_portfolio_defaults_cash_and_targets_when_absent(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text("holdings:\n  AAPL:\n    shares: 10\n")
    out = P.load_portfolio(path)
    assert out["cash"] == 0.0
    assert out["targets"] == {}
    assert out["currency"] is None


def test_load_portfolio_rejects_holding_without_shares_or_value(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text("holdings:\n  AAPL: 10\n")  # missing the {shares:} / {value:} wrapper
    with pytest.raises(ValueError, match="needs a 'shares' or 'value' key"):
        P.load_portfolio(path)


def test_load_portfolio_rejects_non_mapping_holdings(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text("holdings:\n  - AAPL\n  - EQQQ.L\n")
    with pytest.raises(ValueError, match="must map ticker"):
        P.load_portfolio(path)


def _portfolio(cash=0.0, holdings=None, targets=None):
    return {"currency": "USD", "cash": cash, "holdings": holdings or {}, "targets": targets or {}}


def test_snapshot_computes_weights_from_value_holdings_without_pricing(monkeypatch):
    monkeypatch.setattr(D, "spot_price", lambda t: (_ for _ in ()).throw(AssertionError("must not price value holdings")))
    portfolio = _portfolio(cash=1000.0, holdings={
        "AAPL": {"shares": None, "value": 3000.0}, "EQQQ.L": {"shares": None, "value": 6000.0}})
    out = P.snapshot(portfolio)
    assert out["applicable"] is True
    assert out["total_value"] == pytest.approx(10000.0)
    assert out["holdings_pct"] == {"AAPL": 30.0, "EQQQ.L": 60.0}
    assert out["cash_pct"] == pytest.approx(10.0)


def test_snapshot_live_prices_share_holdings(monkeypatch):
    monkeypatch.setattr(D, "spot_price", lambda t: {"AAPL": 100.0}.get(t))
    portfolio = _portfolio(cash=0.0, holdings={"AAPL": {"shares": 50.0, "value": None}})
    out = P.snapshot(portfolio)
    assert out["total_value"] == pytest.approx(5000.0)
    assert out["holdings_pct"] == {"AAPL": 100.0}


def test_snapshot_reuses_caller_supplied_price_for_current_ticker_no_second_fetch(monkeypatch):
    monkeypatch.setattr(D, "spot_price", lambda t: (_ for _ in ()).throw(AssertionError("must reuse caller price")))
    portfolio = _portfolio(holdings={"AAPL": {"shares": 10.0, "value": None}})
    out = P.snapshot(portfolio, current_ticker="AAPL", current_price=200.0)
    assert out["holdings_pct"] == {"AAPL": 100.0}
    assert out["current_ticker_existing_pct"] == pytest.approx(100.0)


def test_snapshot_current_ticker_existing_pct_zero_when_not_held(monkeypatch):
    monkeypatch.setattr(D, "spot_price", lambda t: 100.0)
    portfolio = _portfolio(cash=1000.0, holdings={"AAPL": {"shares": 10.0, "value": None}})
    out = P.snapshot(portfolio, current_ticker="TSLA", current_price=300.0)
    assert out["current_ticker_existing_pct"] == 0.0


def test_snapshot_current_ticker_existing_pct_none_when_ticker_not_supplied(monkeypatch):
    monkeypatch.setattr(D, "spot_price", lambda t: 100.0)
    portfolio = _portfolio(holdings={"AAPL": {"shares": 10.0, "value": None}})
    out = P.snapshot(portfolio)
    assert out["current_ticker_existing_pct"] is None


def test_snapshot_lists_unpriced_holdings_and_excludes_them_from_total(monkeypatch):
    monkeypatch.setattr(D, "spot_price", lambda t: None if t == "DEAD" else 100.0)
    portfolio = _portfolio(cash=0.0, holdings={
        "AAPL": {"shares": 10.0, "value": None}, "DEAD": {"shares": 5.0, "value": None}})
    out = P.snapshot(portfolio)
    assert out["unpriced_holdings"] == ["DEAD"]
    assert out["total_value"] == pytest.approx(1000.0)  # DEAD excluded entirely, not priced at 0
    assert out["holdings_pct"] == {"AAPL": 100.0}


def test_snapshot_not_applicable_when_nothing_priceable_and_no_cash(monkeypatch):
    monkeypatch.setattr(D, "spot_price", lambda t: None)
    portfolio = _portfolio(cash=0.0, holdings={"DEAD": {"shares": 5.0, "value": None}})
    out = P.snapshot(portfolio)
    assert out == {"applicable": False, "reason": "no priceable holdings and no cash"}


def test_snapshot_concentration_hhi_matches_known_formula():
    # 3 equal-weight positions (33.33% each) -> HHI = 3 * 33.33^2 ~= 3333.3,
    # the textbook "10000/n for n equal positions" case (n=3).
    portfolio = _portfolio(cash=0.0, holdings={
        "A": {"shares": None, "value": 1.0}, "B": {"shares": None, "value": 1.0},
        "C": {"shares": None, "value": 1.0}})
    out = P.snapshot(portfolio)
    assert out["concentration_hhi"] == pytest.approx(3333.3, abs=1.0)


def test_snapshot_concentration_hhi_max_when_fully_concentrated():
    portfolio = _portfolio(cash=0.0, holdings={"A": {"shares": None, "value": 1.0}})
    out = P.snapshot(portfolio)
    assert out["concentration_hhi"] == pytest.approx(10000.0)


def test_snapshot_passes_through_targets_and_currency():
    portfolio = _portfolio(cash=100.0, targets={"AAPL": 20.0, "cash": 10.0})
    out = P.snapshot(portfolio)
    assert out["targets_pct"] == {"AAPL": 20.0, "cash": 10.0}
    assert out["currency"] == "USD"

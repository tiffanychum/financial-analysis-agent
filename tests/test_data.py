"""Tests for the quoteSummary-crumb fallbacks: self-computed beta and SEC
EDGAR sector/industry. These are pure-logic paths that don't need network
access to test - `_ticker`/`_yf_session` calls are monkeypatched.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import data as D


class _FakeHistoryTicker:
    def __init__(self, closes: pd.Series):
        self._closes = closes

    def history(self, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        return pd.DataFrame({"Close": self._closes})


def _weekly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", periods=n, freq="W")


def test_compute_beta_matches_known_linear_relationship(monkeypatch):
    # Construct bench returns, then build stock returns as exactly
    # 1.5x bench returns (zero noise) -> beta should recover ~1.5 exactly,
    # since Cov(1.5x, x)/Var(x) = 1.5 by construction.
    rng = np.random.default_rng(7)
    idx = _weekly_index(60)
    bench_returns = rng.normal(0, 0.02, size=len(idx))
    bench_prices = 100 * np.cumprod(1 + bench_returns)
    stock_returns = 1.5 * bench_returns
    stock_prices = 50 * np.cumprod(1 + stock_returns)

    tickers = {
        "TEST": _FakeHistoryTicker(pd.Series(stock_prices, index=idx)),
        "^GSPC": _FakeHistoryTicker(pd.Series(bench_prices, index=idx)),
    }
    monkeypatch.setattr(D, "_ticker", lambda sym: tickers[sym])

    beta, source = D.compute_beta("TEST")
    assert beta == pytest.approx(1.5, abs=1e-6)
    assert "^GSPC" in source
    assert "quoteSummary beta blocked" in source


def test_compute_beta_picks_benchmark_by_exchange_suffix(monkeypatch):
    seen: list[str] = []

    def fake_ticker(sym: str) -> _FakeHistoryTicker:
        seen.append(sym)
        idx = _weekly_index(30)
        return _FakeHistoryTicker(pd.Series(np.linspace(100, 110, len(idx)), index=idx))

    monkeypatch.setattr(D, "_ticker", fake_ticker)
    D.compute_beta("BARC.L")  # ordinary LSE equity, not in the override table
    assert "^FTSE" in seen


@pytest.mark.parametrize("ticker,expected_bench", [
    ("EQQQ.L", "^NDX"),  # Nasdaq-100 UCITS ETF listed in London
    ("IGLN.L", "GC=F"),  # gold ETC listed in London
])
def test_compute_beta_override_beats_suffix_for_wrapper_etfs(monkeypatch, ticker, expected_bench):
    """Tier 1 fix: EQQQ.L previously got a ^FTSE beta from its .L suffix even
    though it tracks the Nasdaq-100, not the FTSE; same failure mode gave the
    gold ETC IGLN.L an equity benchmark."""
    seen: list[str] = []

    def fake_ticker(sym: str) -> _FakeHistoryTicker:
        seen.append(sym)
        idx = _weekly_index(30)
        return _FakeHistoryTicker(pd.Series(np.linspace(100, 110, len(idx)), index=idx))

    monkeypatch.setattr(D, "_ticker", fake_ticker)
    D.compute_beta(ticker)
    assert expected_bench in seen
    assert "^FTSE" not in seen


def test_compute_beta_returns_none_with_too_few_datapoints(monkeypatch):
    idx = _weekly_index(10)  # below the 26-point floor
    tickers = {
        "TEST": _FakeHistoryTicker(pd.Series(np.linspace(100, 105, len(idx)), index=idx)),
        "^GSPC": _FakeHistoryTicker(pd.Series(np.linspace(200, 210, len(idx)), index=idx)),
    }
    monkeypatch.setattr(D, "_ticker", lambda sym: tickers[sym])

    beta, source = D.compute_beta("TEST")
    assert beta is None
    assert source is None


def test_compute_beta_handles_fetch_failure(monkeypatch):
    def boom(sym: str):
        raise RuntimeError("network down")

    monkeypatch.setattr(D, "_ticker", boom)
    beta, source = D.compute_beta("TEST")
    assert beta is None
    assert source is None


class _FakeNewsTicker:
    def __init__(self, news: list[dict]):
        self._news = news

    @property
    def news(self):
        return self._news


def _news_item(title: str, pub_date: str) -> dict:
    return {"content": {"title": title, "pubDate": pub_date}}


def test_fetch_news_filters_out_stale_headlines(monkeypatch):
    now = datetime.now(timezone.utc)
    fresh = _news_item("Apple announces new product", (now - timedelta(days=1)).isoformat())
    stale = _news_item("Apple old news from last year", (now - timedelta(days=400)).isoformat())
    monkeypatch.setattr(D, "_ticker", lambda sym: _FakeNewsTicker([fresh, stale]))
    out = D.fetch_news("AAPL", company_name="Apple Inc.")
    titles = [item["title"] for item in out]
    assert "Apple announces new product" in titles
    assert "Apple old news from last year" not in titles


def test_fetch_news_filters_out_irrelevant_headlines(monkeypatch):
    # live-observed on AAPL: an unrelated UnitedHealth staffing story showed
    # up in the per-ticker feed with no relatedTickers metadata to filter on
    now = datetime.now(timezone.utc).isoformat()
    on_topic = _news_item("Apple rallies on new product launch", now)
    off_topic = _news_item("How many employees does UnitedHealth have in 2026?", now)
    monkeypatch.setattr(D, "_ticker", lambda sym: _FakeNewsTicker([on_topic, off_topic]))
    out = D.fetch_news("AAPL", company_name="Apple Inc.")
    titles = [item["title"] for item in out]
    assert "Apple rallies on new product launch" in titles
    assert "How many employees does UnitedHealth have in 2026?" not in titles


def test_fetch_news_keeps_everything_without_a_company_name_to_check(monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    unrelated = _news_item("Some unrelated market roundup story", now)
    monkeypatch.setattr(D, "_ticker", lambda sym: _FakeNewsTicker([unrelated]))
    out = D.fetch_news("AAPL")  # no company_name -> best-effort, don't filter
    assert len(out) == 1


def test_fetch_news_respects_limit_after_filtering(monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    items = [_news_item(f"Apple story {i}", now) for i in range(20)]
    monkeypatch.setattr(D, "_ticker", lambda sym: _FakeNewsTicker(items))
    out = D.fetch_news("AAPL", limit=3, company_name="Apple Inc.")
    assert len(out) == 3


@pytest.mark.parametrize("sic,expected", [
    (None, None),
    (6510, "Real Estate"),  # 6500-6553 range
    (6021, "Financial Services"),  # national commercial banks
    (6799, "Financial Services"),  # investment offices, top of division H
    (3571, None),  # electronic computers - outside division H entirely
])
def test_sic_sector_maps_only_finance_and_real_estate(sic, expected):
    assert D._sic_sector(sic) == expected


def test_fetch_sec_profile_returns_none_when_ticker_not_in_cik_map(monkeypatch):
    monkeypatch.setattr(D, "_sec_cik_map", lambda: {})
    sector, industry = D.fetch_sec_profile("EQQQ.L")
    assert sector is None
    assert industry is None


@pytest.mark.parametrize("currency,expected_scale,expected_out", [
    ("GBp", 0.01, "GBP"), ("GBX", 0.01, "GBP"),
    ("USD", 1.0, "USD"), (None, 1.0, None),
])
def test_price_scale_converts_only_pence_currencies(currency, expected_scale, expected_out):
    scale, out = D._price_scale(currency)
    assert scale == expected_scale
    assert out == expected_out


class _FakeFundamentalsTicker:
    """Minimal double covering exactly what fetch_fundamentals touches when
    quoteSummary (.info) succeeds and statements are empty - enough to pin
    the GBX/GBp price-scale conversion without a live network call."""

    def __init__(self, info: dict):
        self._info = info
        self.income_stmt = pd.DataFrame()
        self.balance_sheet = pd.DataFrame()
        self.cashflow = pd.DataFrame()

    @property
    def info(self):
        return self._info

    @property
    def growth_estimates(self):
        return pd.DataFrame()


def test_fetch_fundamentals_converts_gbx_price_and_eps_and_target(monkeypatch):
    info = {"currency": "GBp", "currentPrice": 53050.0, "forwardEps": 120.0,
            "targetMeanPrice": 55000.0, "quoteType": "ETF", "beta": 0.9}
    monkeypatch.setattr(D, "_ticker", lambda sym: _FakeFundamentalsTicker(info))
    monkeypatch.setattr(D, "_sec_cik_map", lambda: {})  # avoid a live SEC lookup
    f = D.fetch_fundamentals("EQQQ.L")
    assert f["currency"] == "GBP"
    assert f["price"] == pytest.approx(530.50)
    assert f["eps_fwd"] == pytest.approx(1.20)
    assert f["analyst_target_mean"] == pytest.approx(550.0)
    assert "converted from GBX/GBp" in f["price_source"]


def test_fetch_fundamentals_leaves_non_pence_currency_unscaled(monkeypatch):
    info = {"currency": "USD", "currentPrice": 89.37, "quoteType": "ETF", "beta": 0.28}
    monkeypatch.setattr(D, "_ticker", lambda sym: _FakeFundamentalsTicker(info))
    monkeypatch.setattr(D, "_sec_cik_map", lambda: {})  # avoid a live SEC lookup
    f = D.fetch_fundamentals("IGLN.L")
    assert f["currency"] == "USD"
    assert f["price"] == pytest.approx(89.37)
    assert f["price_source"] == "yfinance quoteSummary"


# ---------------------------------------------------------------- TTM statements

def _stmt_frame(dates: list[str], rows: dict[str, list[float]]) -> pd.DataFrame:
    """Most-recent-first statement frame, matching yfinance's own column
    order (col 0 = latest)."""
    cols = [pd.Timestamp(d) for d in dates]
    return pd.DataFrame({cols[i]: {name: vals[i] for name, vals in rows.items()}
                         for i in range(len(cols))})


class _FakeTTMTicker:
    """Double covering fetch_fundamentals' TTM path: quarterly_* frames
    carry real multi-quarter data, annual income_stmt/balance_sheet/cashflow
    carry multi-year data - enough to pin "TTM sum for flows, latest quarter
    for balance, last-fiscal-year annual as _prev" without live network."""

    def __init__(self, info, quarterly_inc=None, quarterly_bal=None, quarterly_cf=None,
                annual_inc=None, annual_bal=None, annual_cf=None):
        self._info = info
        self.quarterly_income_stmt = quarterly_inc if quarterly_inc is not None else pd.DataFrame()
        self.quarterly_balance_sheet = quarterly_bal if quarterly_bal is not None else pd.DataFrame()
        self.quarterly_cashflow = quarterly_cf if quarterly_cf is not None else pd.DataFrame()
        self.income_stmt = annual_inc if annual_inc is not None else pd.DataFrame()
        self.balance_sheet = annual_bal if annual_bal is not None else pd.DataFrame()
        self.cashflow = annual_cf if annual_cf is not None else pd.DataFrame()

    @property
    def info(self):
        return self._info

    @property
    def growth_estimates(self):
        return pd.DataFrame()


_TTM_INFO = {"currency": "USD", "currentPrice": 100.0, "quoteType": "EQUITY", "beta": 1.0}
_Q_DATES_4 = ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]


def test_ttm_frame_sums_flow_quarters_and_takes_latest_for_balance():
    flow = _stmt_frame(_Q_DATES_4, {"Total Revenue": [120.0, 110.0, 130.0, 100.0]})
    annual = _stmt_frame(["2025-09-30", "2024-09-30"], {"Total Revenue": [400.0, 380.0]})
    frame, date = D._ttm_frame(flow, annual, is_flow=True)
    assert frame.loc["Total Revenue"].iloc[0] == pytest.approx(460.0)  # summed
    assert frame.loc["Total Revenue"].iloc[1] == pytest.approx(400.0)  # last fiscal year
    assert date == pd.Timestamp("2026-06-30")

    balance = _stmt_frame(_Q_DATES_4, {"Total Assets": [900.0, 880.0, 860.0, 840.0]})
    frame_bal, _ = D._ttm_frame(balance, annual=None, is_flow=False)
    assert frame_bal.loc["Total Assets"].iloc[0] == pytest.approx(900.0)  # latest only, not summed
    assert frame_bal.shape[1] == 1  # no annual passed -> no prev column


def test_ttm_frame_empty_when_fewer_than_four_quarters():
    thin = _stmt_frame(_Q_DATES_4[:3], {"Total Revenue": [120.0, 110.0, 130.0]})
    frame, date = D._ttm_frame(thin, annual=None, is_flow=True)
    assert frame.empty and date is None


def test_ttm_frame_survives_quarter_end_matching_fiscal_year_end():
    # Regression: if the latest quarter IS the fiscal year-end quarter, a
    # dict-keyed-by-date frame would collide and silently drop one side -
    # pd.concat is positional and must keep both values distinct.
    flow = _stmt_frame(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"],
                       {"Total Revenue": [120.0, 110.0, 130.0, 100.0]})
    annual = _stmt_frame(["2026-09-30", "2025-09-30"], {"Total Revenue": [999.0, 380.0]})
    frame, _ = D._ttm_frame(flow, annual, is_flow=True)
    assert frame.loc["Total Revenue"].iloc[0] == pytest.approx(460.0)  # TTM, not lost
    assert frame.loc["Total Revenue"].iloc[1] == pytest.approx(999.0)  # LFY annual, not lost


def test_fetch_fundamentals_prefers_ttm_from_quarterly_when_available(monkeypatch):
    q_inc = _stmt_frame(_Q_DATES_4, {"Total Revenue": [120.0, 110.0, 130.0, 100.0]})  # TTM=460
    q_bal = _stmt_frame(_Q_DATES_4, {"Total Assets": [900.0, 880.0, 860.0, 840.0]})  # latest=900
    q_cf = _stmt_frame(_Q_DATES_4, {"Operating Cash Flow": [40.0, 35.0, 45.0, 30.0]})  # TTM=150
    annual_inc = _stmt_frame(["2025-09-30", "2024-09-30", "2023-09-30"],
                             {"Total Revenue": [400.0, 380.0, 360.0]})
    annual_bal = _stmt_frame(["2025-09-30", "2024-09-30"], {"Total Assets": [840.0, 800.0]})
    annual_cf = _stmt_frame(["2025-09-30", "2024-09-30"], {"Operating Cash Flow": [120.0, 110.0]})
    ticker = _FakeTTMTicker(_TTM_INFO, q_inc, q_bal, q_cf, annual_inc, annual_bal, annual_cf)
    monkeypatch.setattr(D, "_ticker", lambda sym: ticker)
    monkeypatch.setattr(D, "_sec_cik_map", lambda: {})

    f = D.fetch_fundamentals("TEST")
    assert f["statement_basis"] == "TTM (trailing 4 quarters)"
    assert f["statement_date"] == "2026-06-30"
    assert f["revenue"] == pytest.approx(460.0)       # TTM sum of 4 quarters
    assert f["revenue_prev"] == pytest.approx(400.0)  # last full fiscal year
    assert f["total_assets"] == pytest.approx(900.0)  # latest single quarter, not summed
    assert f["cfo"] == pytest.approx(150.0)
    # 5y CAGR must read the real multi-year annual columns, not the
    # 2-column TTM/LFY frame: (400/360)**(1/2) - 1, using cols 0 and 2.
    assert f["revenue_growth_5y"] == pytest.approx((400.0 / 360.0) ** (1 / 2) - 1)


def test_fetch_fundamentals_statement_days_stale_matches_ttm_quarter_end(monkeypatch):
    q_inc = _stmt_frame(_Q_DATES_4, {"Total Revenue": [120.0, 110.0, 130.0, 100.0]})
    q_cf = _stmt_frame(_Q_DATES_4, {"Operating Cash Flow": [40.0, 35.0, 45.0, 30.0]})
    ticker = _FakeTTMTicker(_TTM_INFO, q_inc, pd.DataFrame(), q_cf,
                            pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(D, "_ticker", lambda sym: ticker)
    monkeypatch.setattr(D, "_sec_cik_map", lambda: {})

    f = D.fetch_fundamentals("TEST")
    expected = (datetime.now(timezone.utc).date() - datetime(2026, 6, 30).date()).days
    assert f["statement_days_stale"] == expected


def test_fetch_fundamentals_falls_back_to_annual_when_quarterly_too_thin(monkeypatch):
    q_inc = _stmt_frame(_Q_DATES_4[:3], {"Total Revenue": [120.0, 110.0, 130.0]})  # only 3
    annual_inc = _stmt_frame(["2025-09-30", "2024-09-30"], {"Total Revenue": [400.0, 380.0]})
    ticker = _FakeTTMTicker(_TTM_INFO, q_inc, pd.DataFrame(), pd.DataFrame(),
                            annual_inc, pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(D, "_ticker", lambda sym: ticker)
    monkeypatch.setattr(D, "_sec_cik_map", lambda: {})

    f = D.fetch_fundamentals("TEST")
    assert f["statement_basis"] == "annual (fiscal year)"
    assert f["revenue"] == pytest.approx(400.0)
    assert f["statement_date"] == "2025-09-30"


def test_fetch_fundamentals_ttm_income_falls_back_to_annual_balance_sheet(monkeypatch):
    # Quarterly income/cashflow resolve, but quarterly balance sheet is too
    # thin - basis label must say so rather than silently claiming a full
    # TTM read while the balance-sheet fields are actually last-annual.
    q_inc = _stmt_frame(_Q_DATES_4, {"Total Revenue": [120.0, 110.0, 130.0, 100.0]})
    q_cf = _stmt_frame(_Q_DATES_4, {"Operating Cash Flow": [40.0, 35.0, 45.0, 30.0]})
    annual_bal = _stmt_frame(["2025-09-30", "2024-09-30"], {"Total Assets": [840.0, 800.0]})
    ticker = _FakeTTMTicker(_TTM_INFO, q_inc, pd.DataFrame(), q_cf,
                            pd.DataFrame(), annual_bal, pd.DataFrame())
    monkeypatch.setattr(D, "_ticker", lambda sym: ticker)
    monkeypatch.setattr(D, "_sec_cik_map", lambda: {})

    f = D.fetch_fundamentals("TEST")
    assert f["statement_basis"] == "TTM income/cashflow, latest annual balance sheet"
    assert f["revenue"] == pytest.approx(460.0)
    assert f["total_assets"] == pytest.approx(840.0)


# ---------------------------------------------------------------- SEC XBRL

def _xbrl_rec(start: str, end: str, val: float, fy: int = 2025, fp: str = "FY",
              form: str = "10-K") -> dict:
    return {"start": start, "end": end, "val": val, "fy": fy, "fp": fp, "form": form}


def _xbrl_facts(revenue: list[dict] | None = None, net_income: list[dict] | None = None,
                shares: list[dict] | None = None) -> dict:
    return {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax":
                {"units": {"USD": revenue or []}},
            "NetIncomeLoss": {"units": {"USD": net_income or []}},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares or []}}},
    }


# Two fiscal years of discrete quarters (fy/fp deliberately all "2025"/"FY"
# for every record, mirroring the real bug: a 10-K tags every comparative
# income-statement column with the SAME (fy, fp) tuple, so a correct
# implementation must key on (start, end), not (fy, fp)).
_FY2023_Q1 = _xbrl_rec("2023-01-01", "2023-03-31", 180.0, fp="Q1")
_FY2023_Q2 = _xbrl_rec("2023-04-01", "2023-06-30", 220.0, fp="Q2")
_FY2023_Q3 = _xbrl_rec("2023-07-01", "2023-09-29", 230.0, fp="Q3")
_FY2023_ANNUAL = _xbrl_rec("2023-01-01", "2023-12-30", 900.0)  # Q4 = 270 (derived)
_FY2024_Q1 = _xbrl_rec("2024-01-01", "2024-03-31", 200.0, fp="Q1")
_FY2024_Q2 = _xbrl_rec("2024-04-01", "2024-06-30", 250.0, fp="Q2")
_FY2024_Q2_YTD_DUP = _xbrl_rec("2024-01-01", "2024-06-30", 450.0, fp="Q2")  # cumulative dup
_FY2024_Q3 = _xbrl_rec("2024-07-01", "2024-09-29", 270.0, fp="Q3")
_FY2024_Q3_YTD_DUP = _xbrl_rec("2024-01-01", "2024-09-29", 720.0, fp="Q3")  # cumulative dup
_FY2024_ANNUAL = _xbrl_rec("2024-01-01", "2024-12-30", 1000.0)  # Q4 = 280 (derived)
_TWO_FY_QUARTERS = [_FY2023_Q1, _FY2023_Q2, _FY2023_Q3, _FY2023_ANNUAL,
                    _FY2024_Q1, _FY2024_Q2, _FY2024_Q2_YTD_DUP,
                    _FY2024_Q3, _FY2024_Q3_YTD_DUP, _FY2024_ANNUAL]


def test_sec_by_duration_buckets_by_length_and_ignores_ytd_cumulative():
    quarterly, annual = D._sec_by_duration(_xbrl_facts(revenue=_TWO_FY_QUARTERS),
                                           D._SEC_REVENUE_TAGS)
    assert quarterly[("2024-04-01", "2024-06-30")] == pytest.approx(250.0)  # discrete
    assert ("2024-01-01", "2024-06-30") not in quarterly  # 181-day YTD dup excluded
    assert annual[("2024-01-01", "2024-12-30")] == pytest.approx(1000.0)
    assert len(quarterly) == 6  # 3 discrete quarters x 2 fiscal years, dups dropped


def test_sec_ttm_derives_q4_by_subtracting_discrete_quarters_from_annual():
    ttm, ttm_prev, source = D._sec_ttm(_xbrl_facts(revenue=_TWO_FY_QUARTERS), D._SEC_REVENUE_TAGS)
    assert ttm == pytest.approx(200.0 + 250.0 + 270.0 + 280.0)      # FY2024 Q1-Q4 (Q4 derived)
    assert ttm_prev == pytest.approx(180.0 + 220.0 + 230.0 + 270.0)  # FY2023 Q1-Q4 (Q4 derived)
    assert "2024-12-30" in source


def test_sec_ttm_history_returns_every_rolling_window_not_just_latest():
    windows = D.sec_ttm_history(_xbrl_facts(revenue=_TWO_FY_QUARTERS), D._SEC_REVENUE_TAGS)
    # 8 discrete quarters (Q4s derived) -> 5 rolling 4-quarter windows:
    # [Q1..Q4]-23=900, then each window rolls forward one quarter at a time.
    assert [round(v) for _, v in windows] == [900, 920, 950, 990, 1000]
    assert windows[-1][0] == "2024-12-30"  # matches _sec_ttm's own "latest" date


def test_sec_ttm_none_when_fewer_than_four_quarters_resolve():
    ttm, ttm_prev, source = D._sec_ttm(
        _xbrl_facts(revenue=[_FY2024_Q1, _FY2024_Q2, _FY2024_Q3]), D._SEC_REVENUE_TAGS)
    assert (ttm, ttm_prev, source) == (None, None, None)


def test_sec_latest_shares_picks_most_recent_end_date():
    shares = [{"end": "2025-04-17", "val": 100.0}, {"end": "2025-07-17", "val": 95.0},
             {"end": "2025-01-16", "val": 105.0}]
    val, source = D._sec_latest_shares(_xbrl_facts(shares=shares))
    assert val == pytest.approx(95.0)
    assert "2025-07-17" in source


def test_fetch_sec_xbrl_fundamentals_empty_for_unresolved_cik(monkeypatch):
    monkeypatch.setattr(D, "_sec_cik_map", lambda: {})
    assert D.fetch_sec_xbrl_fundamentals("NOPE") == {}


def test_warn_appends_instead_of_overwriting():
    f: dict = {}
    D._warn(f, "first issue")
    D._warn(f, "second issue")
    assert f["data_warning"] == "first issue; second issue"


def test_apply_sec_xbrl_primary_overrides_yahoo_and_labels_basis(monkeypatch):
    xbrl = {"revenue": 1000.0, "revenue_source": "SEC XBRL test",
           "shares": 50.0, "shares_source": "SEC XBRL test"}
    monkeypatch.setattr(D, "fetch_sec_xbrl_fundamentals", lambda ticker: xbrl)
    f = {"revenue": 1000.0, "shares": 48.0, "statement_basis": "TTM (trailing 4 quarters)"}
    D._apply_sec_xbrl_primary(f, "AAPL")
    assert f["revenue"] == pytest.approx(1000.0) and f["revenue_source"] == "SEC XBRL test"
    assert f["shares"] == pytest.approx(50.0) and f["shares_source"] == "SEC XBRL test"
    assert f["statement_basis"] == "TTM (trailing 4 quarters) + SEC XBRL primary (revenue/shares)"


def test_apply_sec_xbrl_primary_flags_disagreement_over_threshold(monkeypatch):
    xbrl = {"revenue": 1200.0, "revenue_source": "SEC XBRL test"}
    monkeypatch.setattr(D, "fetch_sec_xbrl_fundamentals", lambda ticker: xbrl)
    f = {"revenue": 1000.0, "statement_basis": "annual (fiscal year)"}  # 20% off
    D._apply_sec_xbrl_primary(f, "AAPL")
    assert f["revenue"] == pytest.approx(1200.0)  # SEC still wins as primary
    assert "disagree by 20%" in f["data_warning"]


def test_apply_sec_xbrl_primary_noop_when_xbrl_unavailable(monkeypatch):
    monkeypatch.setattr(D, "fetch_sec_xbrl_fundamentals", lambda ticker: {})
    f = {"revenue": 1000.0, "statement_basis": "annual (fiscal year)"}
    D._apply_sec_xbrl_primary(f, "IGLN.L")
    assert f == {"revenue": 1000.0, "statement_basis": "annual (fiscal year)"}


def test_fetch_fundamentals_backtest_snapshot_never_uses_ttm(monkeypatch):
    # col>0 (backtest.py's historical reconstruction) must always read
    # annual columns, even when a full quarterly TTM frame is available.
    q_inc = _stmt_frame(_Q_DATES_4, {"Total Revenue": [120.0, 110.0, 130.0, 100.0]})
    annual_inc = _stmt_frame(["2025-09-30", "2024-09-30", "2023-09-30"],
                             {"Total Revenue": [400.0, 380.0, 360.0]})
    ticker = _FakeTTMTicker(_TTM_INFO, q_inc, pd.DataFrame(), pd.DataFrame(),
                            annual_inc, pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(D, "_ticker", lambda sym: ticker)
    monkeypatch.setattr(D, "_sec_cik_map", lambda: {})

    f = D.fetch_fundamentals("TEST", col=1)
    assert f["statement_basis"] == "annual (fiscal year)"
    assert f["revenue"] == pytest.approx(380.0)  # annual col 1, not any TTM window
    assert f["statement_date"] == "2024-09-30"
    assert "statement_days_stale" not in f


def test_fetch_sec_profile_maps_sic_to_sector_and_keeps_raw_industry(monkeypatch):
    class _Resp:
        def json(self):
            return {"sic": "6022", "sicDescription": "State Commercial Banks"}

    monkeypatch.setattr(D, "_sec_cik_map", lambda: {"JPM": 19617})
    monkeypatch.setattr(D, "_yf_session", lambda: type(
        "S", (), {"get": staticmethod(lambda url, timeout, headers: _Resp())})())

    sector, industry = D.fetch_sec_profile("JPM")
    assert sector == "Financial Services"
    assert industry == "State Commercial Banks"


class _FakeFastInfoTicker:
    def __init__(self, last_price: float | None, currency: str | None):
        self.fast_info = {"lastPrice": last_price, "currency": currency}


def test_spot_price_returns_scaled_price(monkeypatch):
    # GBp/pence: same live-checked scale as _price_scale's own docstring.
    monkeypatch.setattr(D, "_ticker", lambda ticker: _FakeFastInfoTicker(53000.0, "GBp"))
    assert D.spot_price("EQQQ.L") == pytest.approx(530.0)


def test_spot_price_passes_through_non_pence_currency_unscaled(monkeypatch):
    monkeypatch.setattr(D, "_ticker", lambda ticker: _FakeFastInfoTicker(230.5, "USD"))
    assert D.spot_price("AAPL") == pytest.approx(230.5)


def test_spot_price_none_when_fast_info_has_no_price(monkeypatch):
    monkeypatch.setattr(D, "_ticker", lambda ticker: _FakeFastInfoTicker(None, "USD"))
    assert D.spot_price("AAPL") is None


def test_spot_price_none_on_fast_info_exception(monkeypatch):
    class _RaisingTicker:
        @property
        def fast_info(self):
            raise RuntimeError("network down")

    monkeypatch.setattr(D, "_ticker", lambda ticker: _RaisingTicker())
    assert D.spot_price("AAPL") is None


class _FakeEarningsTicker:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_earnings_dates(self, limit: int):
        return self._df


def _earnings_df(rows: list[tuple[str, float | None, float | None, float | None]]) -> pd.DataFrame:
    idx = pd.to_datetime([r[0] for r in rows]).tz_localize("America/New_York")
    return pd.DataFrame({"EPS Estimate": [r[1] for r in rows],
                        "Reported EPS": [r[2] for r in rows],
                        "Surprise(%)": [r[3] for r in rows]}, index=idx).rename_axis("Earnings Date")


def test_fetch_earnings_calendar_reports_next_date_and_recent_surprises(monkeypatch):
    future_date = (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat()
    rows = [
        (future_date, 1.98, None, None),
        ("2025-07-30", 1.89, 2.02, 6.74),
        ("2025-04-30", 1.94, 2.01, 3.46),
        ("2025-01-29", 2.67, 2.84, 6.34),
        ("2024-10-30", 1.77, 1.85, 4.52),
    ]
    monkeypatch.setattr(D, "_ticker", lambda ticker: _FakeEarningsTicker(_earnings_df(rows)))
    out = D.fetch_earnings_calendar("AAPL")
    assert out["next_earnings_date"] == future_date
    assert out["next_earnings_eps_estimate"] == pytest.approx(1.98)
    assert 9 <= out["next_earnings_days_away"] <= 10  # UTC-vs-local date rounding
    assert out["flag_earnings_within_2w"] is True
    assert len(out["recent_earnings_surprises"]) == 4  # capped, matches limit=6 minus the 1 upcoming row
    assert out["recent_earnings_surprises"][0] == {
        "date": "2025-07-30", "eps_estimate": pytest.approx(1.89),
        "eps_actual": pytest.approx(2.02), "surprise_pct": pytest.approx(6.74)}
    assert out["flag_recent_earnings_misses"] is False  # both recent surprises positive


def test_fetch_earnings_calendar_flags_two_consecutive_misses(monkeypatch):
    rows = [("2025-07-30", 1.89, 1.70, -10.05), ("2025-04-30", 1.94, 1.80, -7.22)]
    monkeypatch.setattr(D, "_ticker", lambda ticker: _FakeEarningsTicker(_earnings_df(rows)))
    out = D.fetch_earnings_calendar("AAPL")
    assert out["flag_recent_earnings_misses"] is True


def test_fetch_earnings_calendar_no_flag_within_2w_when_next_earnings_is_far_out(monkeypatch):
    far_date = (datetime.now(timezone.utc) + timedelta(days=60)).date().isoformat()
    rows = [(far_date, 2.0, None, None)]
    monkeypatch.setattr(D, "_ticker", lambda ticker: _FakeEarningsTicker(_earnings_df(rows)))
    out = D.fetch_earnings_calendar("AAPL")
    assert out["flag_earnings_within_2w"] is False


def test_fetch_earnings_calendar_empty_when_no_rows(monkeypatch):
    monkeypatch.setattr(D, "_ticker", lambda ticker: _FakeEarningsTicker(pd.DataFrame()))
    assert D.fetch_earnings_calendar("AAPL") == {}


def test_fetch_earnings_calendar_reports_error_instead_of_raising(monkeypatch):
    class _RaisingTicker:
        def get_earnings_dates(self, limit: int):
            raise ValueError("Missing optional dependency 'lxml'")

    monkeypatch.setattr(D, "_ticker", lambda ticker: _RaisingTicker())
    out = D.fetch_earnings_calendar("AAPL")
    assert out["error"].startswith("ValueError")


@pytest.mark.parametrize("ticker,expected", [
    ("GC=F", "gold"), ("gc=f", "gold"), ("GLD", "gold"), ("IAU", "gold"),
    ("IGLN.L", "gold"), ("SGLN.L", "gold"),
    ("QQQ", "tech"), ("EQQQ.L", "tech"), ("AAPL", "tech"),
])
def test_macro_asset_class_routes_gold_products_to_the_gold_tilt(ticker, expected):
    assert D.macro_asset_class(ticker) == expected

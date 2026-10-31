"""Own-history percentile anchors: is the CURRENT valuation multiple / market-
implied growth rate expensive or cheap relative to THIS ticker's own history
— not a fade model's single fair-value point estimate (DESIGN.md's Layer 2
models are explicitly reviewed as overstating "overvalued" on quality
compounders; this gives the report a second, model-free anchor). Every
number here is real and sourced (SEC XBRL rolling-TTM revenue/FCF + Yahoo
daily price + SEC dei share counts), percentile-ranked against itself; no
LLM math. US equities only (needs a resolvable SEC CIK, see
data.fetch_sec_companyfacts); ETFs get a price-percentile proxy instead
(see etf_price_percentile).

Named simplification: net debt (for EV) and WACC (for the growth percentile)
are held at TODAY's values across the whole lookback window. Both move far
more slowly than price/revenue/FCF, and a fully time-varying reconstruction
would need several more non-standardized XBRL balance-sheet tags for little
marginal accuracy in a PERCENTILE rank, which only cares about relative
position, not the exact historical dollar value.
"""

from __future__ import annotations

import pandas as pd

import data as D
import valuation as V

MIN_QUARTERS = 8  # 2 years of history - a percentile from less isn't trustworthy
_CFO_TAGS = ("NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
_CAPEX_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment",)


def percentile_rank(current: float | None, history: list[float]) -> float | None:
    """% of historical observations <= current. None with fewer than
    MIN_QUARTERS points rather than a percentile nobody should trust."""
    vals = [v for v in history if v is not None]
    if current is None or len(vals) < MIN_QUARTERS:
        return None
    return sum(1 for v in vals if v <= current) / len(vals) * 100


def _price_series(ticker: str, years_back: int = 10) -> pd.Series:
    hist = D._ticker(ticker).history(period=f"{years_back}y", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    s = hist["Close"].dropna()
    s.index = s.index.tz_localize(None)
    return s


def _shares_series(facts: dict) -> pd.Series:
    """Every historical cover-page share-count fact (dei taxonomy) - reused
    for market cap at each past quarter-end, not just today's."""
    recs = facts.get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    rows = {r["end"]: v for r in recs if r.get("end") and (v := D._f(r.get("val"))) is not None}
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series({pd.Timestamp(d): v for d, v in rows.items()}).sort_index()


def _market_cap_at(price: pd.Series, shares: pd.Series, end_date: str) -> float | None:
    """Price on/before end_date x the SEC-reported share count on/before
    end_date - a real historical market cap, not price alone."""
    ts = pd.Timestamp(end_date)
    p_idx, s_idx = price.index.asof(ts), shares.index.asof(ts)
    if pd.isna(p_idx) or pd.isna(s_idx):
        return None
    return float(price.loc[p_idx] * shares.loc[s_idx])


def _multiple_series(ticker: str, facts: dict, denom_history: list[tuple[str, float]],
                     extra_numerator: float = 0.0) -> tuple[list[str], list[float]]:
    """(dates, multiple) at each denom_history quarter-end: numerator is
    market cap as-of that date (+ extra_numerator, e.g. today's net debt for
    an EV-based multiple)."""
    price, shares = _price_series(ticker), _shares_series(facts)
    if price.empty or shares.empty:
        return [], []
    dates, multiples = [], []
    for end_date, denom in denom_history:
        if not denom or denom <= 0:
            continue
        mcap = _market_cap_at(price, shares, end_date)
        if mcap is None:
            continue
        dates.append(end_date)
        multiples.append((mcap + extra_numerator) / denom)
    return dates, multiples


def _result(dates: list[str], hist: list[float], current: float | None, source: str) -> dict:
    pct = percentile_rank(current, hist)
    if pct is None:
        return {"applicable": False,
               "reason": f"only {len(hist)} historical quarters resolved (need >= {MIN_QUARTERS})"}
    return {"applicable": True, "value": current, "percentile": round(pct, 1), "n": len(hist),
           "years_covered": round((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365, 1),
           "source": source}


def _fcf_ttm_history(facts: dict) -> list[tuple[str, float]]:
    cfo = dict(D.sec_ttm_history(facts, _CFO_TAGS))
    capex = dict(D.sec_ttm_history(facts, _CAPEX_TAGS))
    return sorted((d, cfo[d] - capex[d]) for d in cfo if d in capex)


def ps_percentile(ticker: str, f: dict) -> dict:
    """Price/Sales percentile vs this ticker's own SEC XBRL history."""
    facts_doc = D.fetch_sec_companyfacts(ticker)
    if facts_doc is None:
        return {"applicable": False, "reason": "no resolvable US SEC CIK"}
    facts = facts_doc.get("facts", {})
    revenue_history = D.sec_ttm_history(facts, D._SEC_REVENUE_TAGS)
    dates, hist = _multiple_series(ticker, facts, revenue_history)
    mcap, revenue = f.get("market_cap"), f.get("revenue")
    if not mcap or not revenue or revenue <= 0:
        return {"applicable": False, "reason": "missing current market cap or revenue"}
    return _result(dates, hist, mcap / revenue,
                  "SEC XBRL TTM revenue vs Yahoo daily price x SEC dei shares, own 10y history")


def ev_fcf_percentile(ticker: str, f: dict) -> dict:
    """EV/FCF percentile vs this ticker's own SEC XBRL history. Net debt is
    held at today's value across the lookback window (see module docstring)."""
    facts_doc = D.fetch_sec_companyfacts(ticker)
    if facts_doc is None:
        return {"applicable": False, "reason": "no resolvable US SEC CIK"}
    facts = facts_doc.get("facts", {})
    fcf_history = _fcf_ttm_history(facts)
    fcf = f.get("fcf")
    if not fcf or fcf <= 0:
        return {"applicable": False, "reason": "current FCF is missing or non-positive"}
    net_debt = (f.get("total_debt") or 0.0) - (f.get("cash") or 0.0)
    dates, hist = _multiple_series(ticker, facts, fcf_history, extra_numerator=net_debt)
    ev = (f.get("market_cap") or 0.0) + net_debt
    return _result(dates, hist, ev / fcf,
                  "SEC XBRL TTM FCF vs Yahoo daily price x SEC dei shares + today's net debt, own 10y history")


def growth_percentile(ticker: str, f: dict, rf: float, erp: float, current_growth: float | None) -> dict:
    """Percentile of the market-implied growth rate vs this ticker's own
    history: valuation.solve_implied_growth re-run at every past quarter
    using that quarter's real FCF/market-cap but TODAY's WACC held fixed
    (see module docstring). current_growth is the caller's already-solved
    valuation.reverse_dcf()["details"]["implied_growth"] for today."""
    facts_doc = D.fetch_sec_companyfacts(ticker)
    if facts_doc is None:
        return {"applicable": False, "reason": "no resolvable US SEC CIK"}
    facts = facts_doc.get("facts", {})
    fcf_history = _fcf_ttm_history(facts)
    if not fcf_history:
        return {"applicable": False, "reason": "no historical FCF (CFO/CapEx) resolved from XBRL"}
    net_debt = (f.get("total_debt") or 0.0) - (f.get("cash") or 0.0)
    r = V.wacc(f, rf, erp)
    price, shares = _price_series(ticker), _shares_series(facts)
    dates, hist = [], []
    for end_date, fcf_t in fcf_history:
        if fcf_t is None or fcf_t <= 0:
            continue
        mcap = _market_cap_at(price, shares, end_date)
        if mcap is None:
            continue
        g, _resid, reason = V.solve_implied_growth(fcf_t, r, mcap + net_debt)
        if g is None or reason is not None:
            continue
        dates.append(end_date)
        hist.append(g)
    return _result(dates, hist, current_growth,
                  "valuation.solve_implied_growth re-run per historical quarter, today's WACC held fixed")


def etf_price_percentile(ticker: str, years_back: int = 10) -> dict:
    """Own-history price percentile for an ETF/fund - the closest free proxy
    to a forward-P/E percentile for an index tracker. No free historical
    forward-P/E series exists for a UCITS/US index ETF (that is a paid
    data-vendor product), so this uses the fund's own price range instead:
    still real, sourced, own-history, and answers the same practical
    question ("cheap or expensive vs its own range") - a named tradeoff, not
    a claim that price alone captures valuation the way a P/E would."""
    price = _price_series(ticker, years_back)
    if len(price) < MIN_QUARTERS * 20:  # ~20 trading days/month * 2y minimum
        return {"applicable": False, "reason": "insufficient price history"}
    current = float(price.iloc[-1])
    pct = percentile_rank(current, price.tolist())
    if pct is None:
        return {"applicable": False, "reason": "insufficient price history"}
    return {"applicable": True, "value": current, "percentile": round(pct, 1), "n": len(price),
           "years_covered": round((price.index[-1] - price.index[0]).days / 365, 1),
           "source": "Yahoo daily price, own trailing range (forward-P/E percentile proxy - "
                     "no free historical index forward-P/E series exists)"}

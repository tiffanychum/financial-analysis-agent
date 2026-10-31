"""Tier 2: the historical walk-forward harness the module docstring in
valuation.py cites (Penman & Sougiannis 1998; Francis, Olsson & Oswald 2000)
but had never actually run against its own output.

Scope and honest limits (see finance-valuation-accuracy canvas, Tier 2):
- yfinance's annual statements typically expose only ~4 columns, so this can
  walk back 1-3 fiscal years per ticker, not the multi-decade samples the
  academic studies use. Treat results as a small, noisy pilot, not proof.
- Point-in-time fundamentals reuse data.fetch_fundamentals(ticker, col=N) so
  the field mapping is never duplicated; price/shares/market_cap for col>0
  are reconstructed here from historical closes since yfinance's `info` only
  has today's snapshot.
- rf/erp for past years are NOT re-derived (Treasury/Damodaran history would
  need separate scraping); this harness uses the live rf/erp for every past
  year, which is itself an approximation the results should be read through.
- calibrate_weights_from_backtest / calibrate_fade_from_backtest are the
  MECHANISM Tier 2 asked for. They do not overwrite valuation.WEIGHTS or the
  fade floor themselves: this repo's policy is "no guesses" for constants
  that steer every report, and a handful of noisy backtest years is not
  strong enough evidence to responsibly replace 0.5/0.5/0.3/0.2. Re-run with
  a larger ticker basket and feed the result back manually once it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import data as D
import valuation as V


@dataclass
class BacktestRecord:
    ticker: str
    statement_date: str | None
    price_at_valuation: float
    price_after: float | None
    realized_return_pct: float | None
    fair_value: float | None
    upside_pct: float | None
    model_values: dict = field(default_factory=dict)
    direction_hit: bool | None = None


def _historical_snapshot(ticker: str, col: int) -> dict | None:
    """Point-in-time fundamentals for annual column `col` (0=latest). Returns
    None (never a guess) when the statement date or a matching price bar is
    unavailable, mirroring the rest of the engine's no-silent-degradation rule."""
    f = D.fetch_fundamentals(ticker, col=col)
    if not f.get("statement_date") or not f.get("shares"):
        return None
    hist = D._ticker(ticker).history(start=f["statement_date"], period="5d", auto_adjust=True)
    if hist is None or hist.empty:
        return None
    price = D._f(hist["Close"].iloc[0])
    if not price:
        return None
    f["price"] = price
    f["market_cap"] = price * f["shares"]
    f["eps_fwd"] = f.get("net_income", 0) / f["shares"] if f.get("net_income") else None
    f["growth_est"] = f.get("revenue_growth_5y")
    f["growth_est_source"] = "backtest approximation: revenue_growth_5y reused as growth_est"
    return f


def _price_after(ticker: str, statement_date: str, hold_days: int) -> float | None:
    import datetime as dt

    start = dt.date.fromisoformat(statement_date) + dt.timedelta(days=hold_days)
    hist = D._ticker(ticker).history(
        start=start.isoformat(), end=(start + dt.timedelta(days=10)).isoformat(), auto_adjust=True)
    if hist is None or hist.empty:
        return None
    return D._f(hist["Close"].iloc[0])


def run_backtest(tickers: list[str], max_years_back: int = 3,
                 hold_days: int = 365, rf: float = V.RF_DEFAULT, erp: float = V.ERP) -> list[BacktestRecord]:
    """Walk-forward: for each ticker and each available past annual column,
    value the firm as of that statement date and check the realized forward
    return `hold_days` later against the sign/magnitude of upside_pct. Any
    per-ticker/year failure is skipped, never fatal to the whole run."""
    records: list[BacktestRecord] = []
    for ticker in tickers:
        for col in range(1, max_years_back + 1):
            try:
                f = _historical_snapshot(ticker, col)
                if f is None:
                    continue
                out = V.run_valuation(f, rf=rf, erp=erp)
                price_after = _price_after(ticker, f["statement_date"], hold_days)
                realized = ((price_after - f["price"]) / f["price"] * 100
                            if price_after else None)
                hit = (None if realized is None or out["upside_pct"] is None
                       else (realized > 0) == (out["upside_pct"] > 0))
                records.append(BacktestRecord(
                    ticker=ticker, statement_date=f["statement_date"],
                    price_at_valuation=f["price"], price_after=price_after,
                    realized_return_pct=realized, fair_value=out["fair_value"],
                    upside_pct=out["upside_pct"],
                    model_values={n: m["value"] for n, m in out["models"].items()
                                 if m["applicable"] and m["value"] is not None},
                    direction_hit=hit))
            except Exception:  # noqa: BLE001 - one bad ticker/year must not kill the batch
                continue
    return records


def summarize_backtest(records: list[BacktestRecord]) -> dict:
    """Hit rate (direction only) + sample size. Not a Brier score: upside_pct
    is a point estimate, not a probability, so a proper Brier score would
    need a calibrated probability mapping this harness doesn't build yet."""
    scored = [r for r in records if r.direction_hit is not None]
    if not scored:
        return {"n": 0, "hit_rate": None, "note": "no scoreable records"}
    hits = sum(1 for r in scored if r.direction_hit)
    return {"n": len(scored), "hit_rate": hits / len(scored),
            "tickers": sorted({r.ticker for r in scored})}


def calibrate_weights_from_backtest(records: list[BacktestRecord]) -> dict:
    """Inverse-error weighting: models with lower historical absolute pricing
    error (vs price_after) get more weight. Returns a candidate weights dict
    for manual review — does not touch valuation.WEIGHTS (see module docstring)."""
    errors: dict[str, list[float]] = {}
    for r in records:
        if not r.price_after:
            continue
        for name, val in r.model_values.items():
            if name not in V.WEIGHTS:
                continue
            errors.setdefault(name, []).append(abs(val - r.price_after) / r.price_after)
    if not errors:
        return {"weights": None, "n": 0, "note": "no records with a realized price to compare"}
    mean_err = {n: sum(v) / len(v) for n, v in errors.items()}
    inv = {n: 1 / e if e > 0 else 1e9 for n, e in mean_err.items()}
    total = sum(inv.values())
    return {"weights": {n: w / total for n, w in inv.items()},
            "mean_abs_error_pct": {n: e * 100 for n, e in mean_err.items()},
            "n": sum(len(v) for v in errors.values()),
            "current_weights": dict(V.WEIGHTS)}


def calibrate_fade_from_backtest(records: list[BacktestRecord],
                                 sectors: dict[str, str] | None = None) -> dict:
    """Rough per-sector RNOA persistence check: compares each ticker's RNOA in
    its most recent snapshot vs its oldest available snapshot to see whether
    the universal 10y-fade-to-25% assumption is too aggressive or too mild for
    that sector. `sectors` maps ticker -> sector label (pass in separately;
    BacktestRecord does not carry sector to keep the record small)."""
    if sectors is None:
        sectors = {}
    by_ticker: dict[str, list[BacktestRecord]] = {}
    for r in records:
        by_ticker.setdefault(r.ticker, []).append(r)
    per_sector: dict[str, list[float]] = {}
    for ticker, recs in by_ticker.items():
        recs = sorted((r for r in recs if r.model_values.get("reoi")), key=lambda r: r.statement_date or "")
        if len(recs) < 2:
            continue
        sector = sectors.get(ticker, "unknown")
        oldest, newest = recs[0], recs[-1]
        if oldest.model_values.get("reoi") and newest.model_values.get("reoi"):
            persistence = newest.model_values["reoi"] / oldest.model_values["reoi"]
            per_sector.setdefault(sector, []).append(persistence)
    return {"note": "pilot only: too few yfinance years for a real fade-curve fit; "
                    "current universal fade_floor=0.25 unchanged",
            "sample_persistence_by_sector": {
                s: sum(v) / len(v) for s, v in per_sector.items()}}

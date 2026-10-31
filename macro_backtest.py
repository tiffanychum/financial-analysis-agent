"""Historical check of the macro tilt: at every past month-end, what regime
did macro_rules.py's own rules assign (tech or gold), and what did the
ticker return over the next `horizon_months`? Grouped by regime and compared
against the all-months baseline, so "headwind months returned +6%" is read
next to "an average month returned +15%" instead of next to zero.

No LLM, no new thresholds: each month's pack is fed to the SAME rule
functions and the SAME regime_and_tilt() the live `macro` command uses, so
a threshold change in macro_rules.py is automatically what gets tested.

Scope (6 of the 8 live angles): policy_path needs a historical fed-funds-
futures curve (no free source), and event_calendar is a timing flag, not a
regime - both are excluded and named in the output. financial_conditions
needs FRED's HY spread, which only exists from 2023-09 (FRED restarted the
series; live-checked), so before then that angle is `na` and coverage is
5/6 - the existing coverage floor handles it.

Three things keep this from flattering itself:
- Publication lag: FRED dates a value by the period it describes, not the
  day it was public (August core PCE is dated Aug 1, released late Sept).
  Each series is shifted by PUBLICATION_LAG_DAYS before slicing, so a
  month-end regime only sees numbers that were out by then.
- Baseline: in a long bull market most months are "hits"; each regime's
  hit rate and average return are printed against all scored months.
- Episodes: neighboring months share most of a 12-month window and regimes
  persist, so n months is not n independent trials. n_episodes counts
  separate periods (short flickers merged, see EPISODE_MERGE_GAP_MONTHS);
  fewer than MIN_EPISODES_TRUSTED is flagged.

Named simplification: values are today's revised FRED vintages, not the
first-published ones (ALFRED vintage history would fix this at much higher
cost). SAHMREALTIME is already real-time by construction.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import data as D
import macro as MC
import macro_rules as MR
from track_accuracy import wilson_interval

FULL_WINDOW_START = "2006-01-31"  # first month-end all five full-history angles resolve (DTWEXBGS starts 2006-01)
RECENT_WINDOW_START = "2015-01-31"  # Fed balance sheet + reverse repo plumbing look like today's from ~2014-15
HISTORY_YEARS = 22  # 2006 start plus the 12-month YoY / 3-month change lookbacks
MIN_EPISODES_TRUSTED = 5
# Labels flicker near rule thresholds (2022 replays as HHmmHmHHHHHm), so a
# break of up to this many months stays in the same episode. Without this,
# one tightening cycle counted as 3 "independent" headwind periods.
EPISODE_MERGE_GAP_MONTHS = 3

# Conservative days from a FRED observation date to its public release.
# Weekly H.4.1 (WALCL/WTREGEN) is Wednesday data released Thursday; daily
# market series land next business day; DTWEXBGS is published weekly;
# monthly PCE is released ~4 weeks after the month ends (obs date = the
# 1st); UNRATE/SAHMREALTIME come out the first Friday after month end.
PUBLICATION_LAG_DAYS = {
    "WALCL": 2, "WTREGEN": 2, "RRPONTSYD": 1, "DFII10": 1, "T10Y2Y": 1,
    "BAMLH0A0HYM2": 1, "DTWEXBGS": 10, "PCEPILFE": 62, "UNRATE": 38,
    "SAHMREALTIME": 38, "^VIX": 1,
}

BACKTESTABLE_ANGLES = ("liquidity", "inflation", "real_yields", "growth_recession",
                       "financial_conditions", "dollar_fx")
EXCLUDED_ANGLES = {
    "policy_path": "needs the historical fed-funds-futures curve (no free source)",
    "event_calendar": "timing flag (days to next FOMC), not a regime",
}


# ---------------------------------------------------------------- data (network, once per run)

def _public_dated(rows: list[tuple[str, float]], lag_days: int, scale: float = 1.0) -> pd.Series:
    s = pd.Series({pd.Timestamp(d): v * scale for d, v in rows}, dtype=float).sort_index()
    s.index = s.index + pd.Timedelta(days=lag_days)
    return s


def _daily_close(symbol: str) -> pd.Series:
    hist = D._ticker(symbol).history(period=f"{HISTORY_YEARS}y", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    close = hist["Close"]
    close.index = close.index.tz_localize(None)
    return close


def _load_series() -> dict[str, pd.Series]:
    """Every series indexed by the date it became public. One FRED request
    per series for the whole history - never one per historical month."""
    out = {}
    for sid, lag in PUBLICATION_LAG_DAYS.items():
        if sid == "^VIX":
            vix = _daily_close(sid)
            vix.index = vix.index + pd.Timedelta(days=lag)
            out[sid] = vix
            continue
        scale = MC.RRP_TO_MILLIONS if sid == "RRPONTSYD" else 1.0
        out[sid] = _public_dated(MC.fred_series(sid, years_back=HISTORY_YEARS), lag, scale)
    return out


# ---------------------------------------------------------------- point-in-time replay (pure)

def _upto(s: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    return s[s.index <= as_of]


def _latest(s: pd.Series, as_of: pd.Timestamp) -> float | None:
    sub = _upto(s, as_of)
    return float(sub.iloc[-1]) if len(sub) else None


def _change(s: pd.Series, as_of: pd.Timestamp, days_back: int) -> tuple[float | None, float | None]:
    """(latest, then) with `then` = nearest value on/before `days_back` days
    before the latest one - macro.fred_change's rule, on a sliced series."""
    sub = _upto(s, as_of)
    if len(sub) < 2:
        return (float(sub.iloc[-1]) if len(sub) else None), None
    target = sub.index[-1] - pd.Timedelta(days=days_back)
    prior = sub.iloc[:-1][sub.index[:-1] <= target]
    return float(sub.iloc[-1]), (float(prior.iloc[-1]) if len(prior) else None)


def _pct_change(s: pd.Series, as_of: pd.Timestamp, days_back: int) -> float | None:
    latest, then = _change(s, as_of, days_back)
    if latest is None or not then:
        return None
    return (latest / then - 1) * 100


def _annualized_pct(s: pd.Series, as_of: pd.Timestamp, months: int) -> float | None:
    latest, then = _change(s, as_of, months * 30)
    if latest is None or then is None or then <= 0:
        return None
    return ((latest / then) ** (12 / months) - 1) * 100


def _net_liquidity_weekly(series: dict[str, pd.Series]) -> pd.Series:
    """Same Sunday-ending weekly grid and formula as macro.net_liquidity_
    series. A week is labelled by its Sunday, after its public dates, so
    slicing at a month-end never sees the in-progress week."""
    weekly = [series[sid].resample("W").last().ffill() for sid in ("WALCL", "WTREGEN", "RRPONTSYD")]
    joined = pd.concat(weekly, axis=1, keys=["w", "t", "r"]).ffill().dropna()
    return joined["w"] - joined["t"] - joined["r"]


def _pack_at(series: dict[str, pd.Series], net_liq: pd.Series, as_of: pd.Timestamp) -> dict:
    """The sub-dicts of macro.build_macro_pack() the six replayable angles
    read, built only from values public on/before `as_of`."""
    nl = _upto(net_liq, as_of)
    nl_chg = (nl.iloc[-1] / nl.iloc[-13] - 1) * 100 if len(nl) > 13 else None
    ry_latest, ry_then = _change(series["DFII10"], as_of, 90)

    def v(sid: str) -> dict:
        return {"value": _latest(series[sid], as_of)}

    return {
        "liquidity": {"net_liquidity": float(nl.iloc[-1]) if len(nl) else None,
                      "net_liquidity_3m_change_pct": nl_chg},
        "inflation": {"core_pce_yoy_pct": _pct_change(series["PCEPILFE"], as_of, 365),
                      "core_pce_3m_annualized_pct": _annualized_pct(series["PCEPILFE"], as_of, 3)},
        "real_yields": {"real_10y": {"value": ry_latest},
                        "real_10y_3m_change_bps": (ry_latest - ry_then) * 100
                        if ry_latest is not None and ry_then is not None else None},
        "growth_recession": {"unemployment_rate": v("UNRATE"), "sahm_rule": v("SAHMREALTIME"),
                             "yield_curve_10y2y": v("T10Y2Y")},
        "financial_conditions": {"high_yield_spread": v("BAMLH0A0HYM2"), "vix": v("^VIX")},
        "dollar_fx": {"broad_dollar_index": v("DTWEXBGS"),
                      "broad_dollar_index_3m_change_pct": _pct_change(series["DTWEXBGS"], as_of, 90)},
    }


def regime_history(series: dict[str, pd.Series], asset_class: str,
                   as_of_dates: pd.DatetimeIndex) -> tuple[pd.Series, dict[str, str]]:
    """(regime label per month-end, first month each angle resolved)."""
    angles = {name: MR.ANGLES[name] for name in BACKTESTABLE_ANGLES}
    net_liq = _net_liquidity_weekly(series)
    labels, first_applicable = {}, {}
    for as_of in as_of_dates:
        card = MR.scorecard(_pack_at(series, net_liq, as_of), angles)
        for name, result in card["angles"].items():
            if result["applicable"]:
                first_applicable.setdefault(name, as_of.strftime("%Y-%m"))
        labels[as_of] = MR.regime_and_tilt(card)[asset_class]["regime"]
    return pd.Series(labels), first_applicable


# ---------------------------------------------------------------- forward returns + stats (pure)

def forward_returns(close: pd.Series, as_of_dates: pd.DatetimeIndex, horizon_months: int) -> pd.Series:
    """Return from the last close on/before each month-end to the last close
    on/before the same date `horizon_months` later. Months whose window
    extends past the last available close are left out (unscored, not a miss)."""
    if close.empty:
        return pd.Series(dtype=float)
    out = {}
    for as_of in as_of_dates:
        end = as_of + pd.DateOffset(months=horizon_months)
        start_px, end_px = _upto(close, as_of), _upto(close, end)
        if end > close.index[-1] or start_px.empty:
            continue
        out[as_of] = float(end_px.iloc[-1] / start_px.iloc[-1] - 1)
    return pd.Series(out, dtype=float)


def _episodes(labels: pd.Series, label: str) -> int:
    """Separate periods of `label` in month order, where a break of up to
    EPISODE_MERGE_GAP_MONTHS other-label months does not start a new one."""
    episodes, last = 0, None
    for i, on in enumerate((labels == label).to_numpy()):
        if not on:
            continue
        if last is None or i - last - 1 > EPISODE_MERGE_GAP_MONTHS:
            episodes += 1
        last = i
    return episodes


def _stats(returns: pd.Series) -> dict:
    n = len(returns)
    hits = int((returns > 0).sum())
    return {"n_months": n, "hit_rate": round(hits / n, 3), "hit_rate_95ci": wilson_interval(hits, n),
            "avg_return_pct": round(float(returns.mean()) * 100, 1)}


def window_stats(labels: pd.Series, returns: pd.Series, start: str) -> dict:
    scored = returns[returns.index >= pd.Timestamp(start)] if not returns.empty else returns
    if scored.empty:
        return {"applicable": False, "reason": f"no month since {start} has a full forward window"}
    window_labels = labels.reindex(scored.index)
    baseline = _stats(scored)
    regimes = {}
    for label in ("supportive", "mixed", "headwind", "unmeasured"):
        rets = scored[window_labels == label]
        if rets.empty:
            continue
        s = _stats(rets)
        s["n_episodes"] = _episodes(window_labels, label)
        s["trusted"] = s["n_episodes"] >= MIN_EPISODES_TRUSTED
        s["vs_baseline_return_pp"] = round(s["avg_return_pct"] - baseline["avg_return_pct"], 1)
        s["vs_baseline_hit_rate_pp"] = round((s["hit_rate"] - baseline["hit_rate"]) * 100, 1)
        regimes[label] = s
    return {"applicable": True, "start": scored.index[0].strftime("%Y-%m"),
            "end": scored.index[-1].strftime("%Y-%m"), "baseline": baseline, "regimes": regimes}


# ---------------------------------------------------------------- orchestration

def tilt_forward_returns(ticker: str, horizon_months: int = 12) -> dict:
    asset_class = D.macro_asset_class(ticker)
    series = _load_series()
    close = _daily_close(ticker)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    as_of_dates = pd.date_range(FULL_WINDOW_START, today, freq="ME")
    labels, first_applicable = regime_history(series, asset_class, as_of_dates)
    returns = forward_returns(close, as_of_dates, horizon_months)
    return {
        "ticker": ticker, "asset_class": asset_class, "horizon_months": horizon_months,
        "price_history_start": close.index[0].strftime("%Y-%m-%d") if not close.empty else None,
        "latest_regime": {"as_of": labels.index[-1].strftime("%Y-%m-%d"), "regime": labels.iloc[-1]},
        "windows": {"full": window_stats(labels, returns, FULL_WINDOW_START),
                    "recent": window_stats(labels, returns, RECENT_WINDOW_START)},
        "angles_used": list(BACKTESTABLE_ANGLES), "angles_excluded": EXCLUDED_ANGLES,
        "angle_first_applicable": first_applicable,
        "min_episodes_trusted": MIN_EPISODES_TRUSTED,
    }

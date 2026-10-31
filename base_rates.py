"""Historical base rates for the technical analyst persona's discrete tape
signals (trend crosses, RSI-zone entries, MACD crosses): the last times
THIS exact signal fired on THIS ticker over its own trailing ~15y, what
fraction of the time it was up over the run's holding period (default 12
months) later, and how many times it
actually fired (n) - a base rate a casual trader can sanity-check a live
signal against, not a backtest recommendation to trade the signal itself.

Signal events are detected on zone/cross ENTRY (this-bar-true,
previous-bar-false), not persistent-day counts - "RSI<30 today" can stay
true for weeks in one downtrend, and counting every one of those days as a
separate "signal" would badly inflate n with one autocorrelated episode.

Price scale (pence vs GBP, see data._price_scale) does not matter here:
every signal (SMA/MACD cross timing, RSI) and the forward return itself is
a ratio/sign computation, invariant to any constant multiplicative rescale
of the whole price series - unlike anchors.py/fetch_technicals, which print
absolute price levels and do need the pence conversion.

Named simplification: hit = forward return > 0 (absolute), not benchmark-
relative like track_accuracy.py's scoring - a signal that merely rode a
long bull market still counts as a "hit" here. The trader/risk seats
already see the macro tilt (macro_rules.py) for that broader context;
duplicating a benchmark-relative version of every one of these six signals
would be a second, heavier feature for a report line meant as a quick
sanity check, not the report's main claim.
"""

from __future__ import annotations

import pandas as pd

import data as D
from track_accuracy import wilson_interval

TRADING_DAYS_PER_MONTH = 21
FORWARD_DAYS = 12 * TRADING_DAYS_PER_MONTH  # default 12-month window (252 trading days)
MIN_EVENTS_TRUSTED = 3  # a 1/1 or 2/2 "hit rate" is not a base rate


def _indicators(c: pd.Series) -> pd.DataFrame:
    """Same formulas as data.fetch_technicals (SMA50/200, RSI14, MACD/
    signal), applied to the whole history instead of just the latest bar."""
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    return pd.DataFrame({"close": c, "sma50": sma50, "sma200": sma200,
                        "rsi14": rsi, "macd": macd, "macd_signal": macd_signal})


def _entries(state: pd.Series) -> pd.Series:
    """True only on the bar a boolean condition turns on (False/NaN the bar
    before) - one row per discrete episode, not one row per day it holds."""
    on = state.astype("boolean").fillna(False).astype(bool)
    return on & ~on.shift(1, fill_value=False)


def _signal_entries(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "golden_cross": _entries(df["sma50"] > df["sma200"]),
        "death_cross": _entries(df["sma50"] < df["sma200"]),
        "rsi_oversold_entry": _entries(df["rsi14"] < 30),
        "rsi_overbought_entry": _entries(df["rsi14"] > 70),
        "macd_bull_cross": _entries(df["macd"] > df["macd_signal"]),
        "macd_bear_cross": _entries(df["macd"] < df["macd_signal"]),
    }


def _hit_rate(close: pd.Series, entries: pd.Series, forward_days: int = FORWARD_DAYS) -> dict:
    fired_at = close.index[entries.to_numpy()]
    fwd_returns = []
    for ts in fired_at:
        pos = close.index.get_loc(ts)
        if pos + forward_days >= len(close):
            continue  # too recent to have a full forward window yet - unscored, not a miss
        fwd_returns.append(float(close.iloc[pos + forward_days] / close.iloc[pos] - 1))
    n = len(fwd_returns)
    if n == 0:
        return {"applicable": False, "n_fired": len(fired_at),
                "reason": "fired 0 times with a full forward window in the lookback"}
    hits = sum(1 for r in fwd_returns if r > 0)
    return {"applicable": True, "n": n, "n_fired_total": len(fired_at),
            "hit_rate": round(hits / n, 3), "hit_rate_95ci": wilson_interval(hits, n),
            "avg_forward_return_pct": round(sum(fwd_returns) / n * 100, 1),
            "trusted": n >= MIN_EVENTS_TRUSTED}


def signal_base_rates(ticker: str, years_back: int = 15, horizon_months: int = 12) -> dict:
    """{signal_name: hit-rate result} for every signal in _signal_entries,
    plus the actual history span used. Runs for any ticker with price
    history - ETFs/commodities included, no fundamentals regime gate. The
    forward window follows the run's holding period (horizon_months)."""
    hist = D._ticker(ticker).history(period=f"{years_back}y", auto_adjust=True)
    if hist is None or hist.empty:
        return {}
    df = _indicators(hist["Close"])
    forward_days = horizon_months * TRADING_DAYS_PER_MONTH
    signals = {name: _hit_rate(df["close"], entries, forward_days)
              for name, entries in _signal_entries(df).items()}
    return {"years_covered": round((hist.index[-1] - hist.index[0]).days / 365, 1),
            "horizon_months": horizon_months, "signals": signals}

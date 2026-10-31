"""Tests for base_rates.py. The indicator formulas (SMA/RSI/MACD) mirror
data.fetch_technicals' already-standard textbook math (untested elsewhere
for the same reason); these tests cover the logic this module actually
owns: entry-only event detection, forward-window hit-rate/Wilson wiring,
and the signal_base_rates orchestration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import base_rates as BR
import data as D


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_entries_fires_only_on_the_transition_bar():
    state = pd.Series([False, True, True, False, True])
    assert BR._entries(state).tolist() == [False, True, False, False, True]


def test_entries_treats_leading_nan_as_false():
    state = pd.Series([np.nan, np.nan, True, True])
    assert BR._entries(state).tolist() == [False, False, True, False]


def test_hit_rate_counts_positive_forward_return_as_a_hit():
    n = BR.FORWARD_DAYS + 10
    close = pd.Series(np.linspace(100.0, 200.0, n), index=_dates(n))  # always rising
    entries = pd.Series(False, index=close.index)
    entries.iloc[5] = True  # one event, well within the forward window
    result = BR._hit_rate(close, entries)
    assert result["applicable"] is True
    assert result["n"] == 1
    assert result["n_fired_total"] == 1
    assert result["hit_rate"] == 1.0  # rising series -> positive forward return
    assert result["avg_forward_return_pct"] == pytest.approx(
        (close.iloc[5 + BR.FORWARD_DAYS] / close.iloc[5] - 1) * 100, abs=0.05)


def test_hit_rate_counts_negative_forward_return_as_a_miss():
    n = BR.FORWARD_DAYS + 10
    close = pd.Series(np.linspace(200.0, 100.0, n), index=_dates(n))  # always falling
    entries = pd.Series(False, index=close.index)
    entries.iloc[5] = True
    result = BR._hit_rate(close, entries)
    assert result["hit_rate"] == 0.0


def test_hit_rate_excludes_events_without_a_full_forward_window(caplog=None):
    n = BR.FORWARD_DAYS + 10
    close = pd.Series(np.linspace(100.0, 200.0, n), index=_dates(n))
    entries = pd.Series(False, index=close.index)
    entries.iloc[5] = True  # scorable
    entries.iloc[-1] = True  # too recent - no forward window left
    result = BR._hit_rate(close, entries)
    assert result["n"] == 1  # only the scorable one counts toward the rate
    assert result["n_fired_total"] == 2  # but both fired, honestly reported


def test_hit_rate_not_applicable_when_every_event_lacks_a_forward_window():
    n = 50
    close = pd.Series(np.linspace(100.0, 110.0, n), index=_dates(n))
    entries = pd.Series(False, index=close.index)
    entries.iloc[-1] = True
    result = BR._hit_rate(close, entries)
    assert result == {"applicable": False, "n_fired": 1,
                      "reason": "fired 0 times with a full forward window in the lookback"}


def test_hit_rate_marks_small_samples_as_untrusted():
    n = BR.FORWARD_DAYS + 10
    close = pd.Series(np.linspace(100.0, 200.0, n), index=_dates(n))
    entries = pd.Series(False, index=close.index)
    entries.iloc[5] = True  # single event -> n=1, below MIN_EVENTS_TRUSTED
    result = BR._hit_rate(close, entries)
    assert result["trusted"] is False


def test_hit_rate_uses_wilson_interval_not_a_raw_proportion():
    n = BR.FORWARD_DAYS + 10
    close = pd.Series(np.linspace(100.0, 200.0, n), index=_dates(n))
    entries = pd.Series(False, index=close.index)
    entries.iloc[5] = True
    result = BR._hit_rate(close, entries)
    from track_accuracy import wilson_interval
    assert result["hit_rate_95ci"] == wilson_interval(1, 1)
    assert result["hit_rate_95ci"][0] < result["hit_rate"]  # wide interval, not a point estimate


def test_hit_rate_uses_the_given_forward_window():
    n = BR.FORWARD_DAYS + 10
    close = pd.Series(np.linspace(100.0, 200.0, n), index=_dates(n))
    entries = pd.Series(False, index=close.index)
    entries.iloc[5] = True
    short = BR._hit_rate(close, entries, forward_days=63)
    assert short["avg_forward_return_pct"] == pytest.approx(
        (close.iloc[5 + 63] / close.iloc[5] - 1) * 100, abs=0.05)


def test_signal_base_rates_reports_and_applies_the_horizon(monkeypatch):
    n = 400
    close = pd.Series(np.linspace(100.0, 200.0, n), index=_dates(n))

    class _FakeTicker:
        def history(self, **kw):
            return pd.DataFrame({"Close": close})

    monkeypatch.setattr(D, "_ticker", lambda ticker: _FakeTicker())
    calls = []
    monkeypatch.setattr(BR, "_hit_rate",
                        lambda c, e, forward_days: calls.append(forward_days) or {"applicable": False})
    result = BR.signal_base_rates("TEST", horizon_months=3)
    assert result["horizon_months"] == 3
    assert set(calls) == {3 * BR.TRADING_DAYS_PER_MONTH}


def test_signal_base_rates_returns_empty_dict_without_price_history(monkeypatch):
    class _EmptyTicker:
        def history(self, **kw):
            return pd.DataFrame()

    monkeypatch.setattr(D, "_ticker", lambda ticker: _EmptyTicker())
    assert BR.signal_base_rates("NEWLISTING") == {}


def test_signal_base_rates_covers_all_six_signals_and_reports_years(monkeypatch):
    n = 900  # ~3.5y of daily bars - enough for SMA200 warmup + a scored cross
    rng = np.random.default_rng(7)
    # a random walk has real crosses/RSI zone visits in both directions,
    # unlike a pure monotonic series (which would make one whole signal
    # family - death cross or golden cross - never fire at all).
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, n)), index=_dates(n))

    class _FakeTicker:
        def history(self, **kw):
            return pd.DataFrame({"Close": close})

    monkeypatch.setattr(D, "_ticker", lambda ticker: _FakeTicker())
    result = BR.signal_base_rates("TEST", years_back=15)
    assert set(result["signals"]) == {"golden_cross", "death_cross", "rsi_oversold_entry",
                                      "rsi_overbought_entry", "macd_bull_cross", "macd_bear_cross"}
    assert result["years_covered"] == pytest.approx(n / 365, abs=0.1)

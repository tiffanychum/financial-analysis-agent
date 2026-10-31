"""Tests for macro_backtest.py: point-in-time replay (publication lag, no
lookahead, same change rule as macro.fred_change), reuse of the live
macro_rules angles, forward-return windows, and the baseline/episode stats.
All fixtures are synthetic - no FRED/Yahoo calls.
"""
from __future__ import annotations

import pandas as pd
import pytest

import macro as MC
import macro_backtest as MB
import macro_rules as MR


def _ts(d: str) -> pd.Timestamp:
    return pd.Timestamp(d)


def test_public_dated_shifts_by_release_lag_and_rescales():
    s = MB._public_dated([("2020-08-01", 2.5)], lag_days=62, scale=1000.0)
    assert s.index[0] == _ts("2020-10-02")
    assert s.iloc[0] == 2500.0


def test_publication_lag_hides_values_not_yet_released():
    # August core PCE (dated Aug 1) is not public at the end of August.
    pce = MB._public_dated([("2020-07-01", 100.0), ("2020-08-01", 101.0)], MB.PUBLICATION_LAG_DAYS["PCEPILFE"])
    assert MB._latest(pce, _ts("2020-08-31")) is None
    assert MB._latest(pce, _ts("2020-09-30")) == 100.0
    assert MB._latest(pce, _ts("2020-10-31")) == 101.0


def test_change_matches_live_fred_change_rule(monkeypatch):
    rows = [(d.strftime("%Y-%m-%d"), float(i)) for i, d in
            enumerate(pd.date_range("2024-01-03", periods=40, freq="7D"))]
    monkeypatch.setattr(MC, "fred_series", lambda series_id, years_back=3: rows)
    live_latest, live_then, _, _ = MC.fred_change("X", 90)
    s = MB._public_dated(rows, lag_days=0)
    assert MB._change(s, s.index[-1], 90) == (live_latest, live_then)


def test_change_ignores_values_after_as_of():
    s = MB._public_dated([("2024-01-01", 1.0), ("2024-04-01", 2.0), ("2024-07-01", 9.0)], lag_days=0)
    assert MB._change(s, _ts("2024-04-15"), 90) == (2.0, 1.0)


def _flat_series() -> dict[str, pd.Series]:
    idx = pd.date_range("2004-01-01", "2008-12-31", freq="7D")
    return {sid: pd.Series(1.0, index=idx) for sid in MB.PUBLICATION_LAG_DAYS}


def test_regime_history_feeds_the_live_rule_functions(monkeypatch):
    # Every replayed angle reads unemployment from the rebuilt pack: low
    # before 2007, high after, so the regime must flip at that point.
    series = _flat_series()
    series["UNRATE"] = pd.Series([4.0, 8.0], index=[_ts("2004-01-01"), _ts("2007-01-01")])

    def fake_angle(pack):
        u = pack["growth_recession"]["unemployment_rate"]["value"]
        impact = "supportive" if u < 5 else "headwind"
        return {"applicable": True, "tech_impact": impact, "gold_impact": "neutral"}

    for name in MB.BACKTESTABLE_ANGLES:
        monkeypatch.setitem(MR.ANGLES, name, fake_angle)
    dates = pd.DatetimeIndex([_ts("2006-06-30"), _ts("2007-06-30")])
    labels, first = MB.regime_history(series, "tech", dates)
    assert labels.tolist() == ["supportive", "headwind"]
    assert first["growth_recession"] == "2006-06"


def test_regime_history_never_uses_excluded_angles(monkeypatch):
    def boom(pack):
        raise AssertionError("excluded angle was evaluated")

    for name in MB.EXCLUDED_ANGLES:
        monkeypatch.setitem(MR.ANGLES, name, boom)
    labels, _ = MB.regime_history(_flat_series(), "tech", pd.DatetimeIndex([_ts("2007-06-30")]))
    assert len(labels) == 1


def test_scorecard_subset_measures_coverage_against_the_subset():
    angles = {name: MR.ANGLES[name] for name in MB.BACKTESTABLE_ANGLES}
    card = MR.scorecard({}, angles)
    assert set(card["angles"]) == set(MB.BACKTESTABLE_ANGLES)
    assert card["coverage_detail"] == "0/6 angles resolved"


def test_forward_returns_uses_last_close_on_or_before_and_drops_incomplete_windows():
    close = pd.Series([100.0, 110.0, 121.0],
                      index=[_ts("2020-01-30"), _ts("2021-01-29"), _ts("2021-06-30")])
    dates = pd.DatetimeIndex([_ts("2020-01-31"), _ts("2021-01-31")])
    out = MB.forward_returns(close, dates, horizon_months=12)
    assert list(out.index) == [_ts("2020-01-31")]  # 2021-01 window runs past the last close
    assert out.iloc[0] == pytest.approx(0.10)


def test_forward_returns_window_follows_horizon_months():
    close = pd.Series([100.0, 105.0, 130.0, 140.0],
                      index=[_ts("2020-01-31"), _ts("2020-04-30"), _ts("2021-01-29"), _ts("2021-03-31")])
    dates = pd.DatetimeIndex([_ts("2020-01-31")])
    assert MB.forward_returns(close, dates, 3).iloc[0] == pytest.approx(0.05)
    assert MB.forward_returns(close, dates, 12).iloc[0] == pytest.approx(0.30)


def test_episodes_merges_short_flickers_into_one_period():
    # 2022's replayed labels: one tightening cycle, not three periods.
    labels = pd.Series(list("HHmmHmHHHHHm"))
    assert MB._episodes(labels, "H") == 1


def test_episodes_splits_on_a_break_longer_than_the_merge_gap():
    gap = MB.EPISODE_MERGE_GAP_MONTHS
    labels = pd.Series(["H"] + ["m"] * (gap + 1) + ["H"])
    assert MB._episodes(labels, "H") == 2
    assert MB._episodes(pd.Series(["H"] + ["m"] * gap + ["H"]), "H") == 1


def test_window_stats_compares_each_regime_with_the_all_months_baseline():
    idx = pd.date_range("2010-01-31", periods=6, freq="ME")
    labels = pd.Series(["supportive", "supportive", "headwind", "headwind", "supportive", "mixed"], index=idx)
    returns = pd.Series([0.20, 0.10, -0.10, 0.00, 0.30, 0.10], index=idx)
    w = MB.window_stats(labels, returns, "2010-01-31")
    assert w["baseline"]["n_months"] == 6
    assert w["baseline"]["avg_return_pct"] == pytest.approx(10.0)
    sup, head = w["regimes"]["supportive"], w["regimes"]["headwind"]
    assert sup["n_months"] == 3 and sup["n_episodes"] == 1  # 2-month break merges
    assert sup["vs_baseline_return_pp"] == pytest.approx(10.0)
    assert head["hit_rate"] == 0.0  # a flat 0% return is not a hit
    assert head["vs_baseline_return_pp"] == pytest.approx(-15.0)
    assert sup["trusted"] is False  # 1 period < MIN_EPISODES_TRUSTED


def test_window_stats_respects_window_start():
    idx = pd.date_range("2014-11-30", periods=4, freq="ME")
    labels = pd.Series("mixed", index=idx)
    returns = pd.Series([0.5, 0.5, 0.1, 0.1], index=idx)
    w = MB.window_stats(labels, returns, MB.RECENT_WINDOW_START)
    assert w["start"] == "2015-01"
    assert w["baseline"]["n_months"] == 2


def test_window_stats_not_applicable_without_scored_months():
    w = MB.window_stats(pd.Series(dtype=object), pd.Series(dtype=float), "2015-01-31")
    assert w["applicable"] is False

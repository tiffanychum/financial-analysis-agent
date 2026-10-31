"""Tests for macro.py (keyless fetchers, pure-logic parsing) and
macro_rules.py (deterministic rules, scorecard/tilt, FOMC redline). No
network access - every fetcher that would hit FRED/federalreserve.gov/Yahoo
is monkeypatched at its own function so the arithmetic/alignment/regex
logic downstream of the fetch is what's actually under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

import macro as MC  # noqa: E402
import macro_rules as MR  # noqa: E402

# ---------------------------------------------------------------- macro.py: parsing

def test_mixed_fraction_to_float_parses_fed_statement_formats():
    assert MC._mixed_fraction_to_float("3-1/2") == 3.5
    assert MC._mixed_fraction_to_float("4") == 4.0
    assert MC._mixed_fraction_to_float("4.25") == 4.25
    assert MC._mixed_fraction_to_float("not-a-number") is None


def test_parse_target_range_reads_mixed_fraction_statement_language():
    text = ("The Committee decided to maintain the target range for the "
           "federal funds rate at 4 to 4-1/4 percent.")
    assert MC.parse_target_range(text) == (4.0, 4.25)


def test_parse_target_range_none_when_phrase_absent():
    assert MC.parse_target_range("No target range language here.") is None
    assert MC.parse_target_range(None) is None


def test_nearest_on_or_before_finds_closest_prior_observation():
    rows = [("2026-01-01", 1.0), ("2026-02-01", 2.0), ("2026-03-01", 3.0)]
    assert MC._nearest_on_or_before(rows, "2026-02-15") == ("2026-02-01", 2.0)
    assert MC._nearest_on_or_before(rows, "2026-01-01") == ("2026-01-01", 1.0)
    assert MC._nearest_on_or_before(rows, "2025-12-01") is None


# ---------------------------------------------------------------- macro.py: futures path

def test_fed_funds_futures_implied_rate_is_100_minus_price(monkeypatch):
    monkeypatch.setattr(MC, "yahoo_price", lambda symbol: (95.5, f"yfinance {symbol}"))
    rate, source = MC.fed_funds_futures_implied_rate()
    assert rate == pytest.approx(4.5)
    assert "ZQ=F" in source


def test_fed_funds_futures_implied_rate_none_when_price_unavailable(monkeypatch):
    monkeypatch.setattr(MC, "yahoo_price", lambda symbol: (None, None))
    rate, source = MC.fed_funds_futures_implied_rate()
    assert rate is None
    assert source is None


# ---------------------------------------------------------------- macro.py: FRED-derived pct changes

def test_fred_yoy_pct_computes_percent_change_over_one_year(monkeypatch):
    rows = [("2025-01-15", 100.0), ("2026-01-15", 103.0)]
    monkeypatch.setattr(MC, "fred_series", lambda series_id, years_back=3: rows)
    pct, date = MC.fred_yoy_pct("PCEPILFE")
    assert pct == pytest.approx(3.0)
    assert date == "2026-01-15"


def test_fred_annualized_pct_compounds_trailing_window_to_a_year(monkeypatch):
    # +1%/mo compounded over 3mo, annualized: (1.01^1)^(12/3) - 1 ~= 4.06%,
    # matches fred_annualized_pct's periods_per_year formula directly.
    rows = [("2025-10-15", 100.0), ("2026-01-15", 101.0)]
    monkeypatch.setattr(MC, "fred_series", lambda series_id, years_back=3: rows)
    pct = MC.fred_annualized_pct("PCEPILFE", months=3)
    assert pct == pytest.approx(((101.0 / 100.0) ** 4 - 1) * 100)


# ---------------------------------------------------------------- macro.py: net liquidity alignment

def _weekly(values: dict[str, float]) -> pd.Series:
    return pd.Series({pd.Timestamp(d): v for d, v in values.items()}).sort_index()


def test_net_liquidity_series_subtracts_tga_and_rrp_in_matching_units(monkeypatch):
    # FRED units: WALCL/WTREGEN in millions, RRPONTSYD in billions.
    walcl = _weekly({"2026-01-04": 7_000_000.0, "2026-01-11": 7_100_000.0})
    tga = _weekly({"2026-01-04": 700_000.0, "2026-01-11": 650_000.0})
    rrp = _weekly({"2026-01-04": 2_500.0, "2026-01-11": 2_000.0})
    series_by_id = {"WALCL": walcl, "WTREGEN": tga, "RRPONTSYD": rrp}
    monkeypatch.setattr(MC, "_weekly_series_from_fred",
                        lambda series_id, years_back=1: series_by_id[series_id])
    net = MC.net_liquidity_series(years_back=1)
    assert net.loc[pd.Timestamp("2026-01-04")] == pytest.approx(7_000_000.0 - 700_000.0 - 2_500_000.0)
    assert net.loc[pd.Timestamp("2026-01-11")] == pytest.approx(7_100_000.0 - 650_000.0 - 2_000_000.0)


def test_series_rescales_reverse_repo_billions_to_millions(monkeypatch):
    monkeypatch.setattr(MC, "fred_latest", lambda sid, years_back=3: (2553.7, "2022-12-30", "FRED RRPONTSYD (2022-12-30)"))
    out = MC._series("reverse_repo", "RRPONTSYD", years_back=1, scale=MC.RRP_TO_MILLIONS)
    assert out["value"] == pytest.approx(2_553_700.0)
    assert "rescaled" in out["source"]


def test_weekly_series_from_yahoo_resamples_to_sunday_ending_weeks(monkeypatch):
    # Regression: an earlier version returned the raw daily series without
    # resampling, so joining it against FRED's Sunday-ending weekly grid in
    # rolling_correlation/net_liquidity_vs_qqq_correlation dropped every row.
    idx = pd.date_range("2026-01-01", periods=10, freq="D")  # Thu..Sat
    closes = pd.Series(range(100, 110), index=idx, dtype=float)

    class _FakeTicker:
        def history(self, period, interval, auto_adjust):
            return pd.DataFrame({"Close": closes})

    monkeypatch.setattr(MC.D, "_ticker", lambda symbol: _FakeTicker())
    s = MC._weekly_series_from_yahoo("QQQ", years_back=1)
    assert not s.empty
    assert all(ts.dayofweek == 6 for ts in s.index)  # 6 == Sunday


def test_net_liquidity_vs_qqq_correlation_recovers_strong_positive_relationship(monkeypatch):
    # A perfectly linear (constant-diff) fixture would give diff()/pct_change()
    # zero variance -> an undefined (NaN) correlation, not a strong one - use
    # noisy weekly returns instead, with net liquidity's diffs exactly
    # proportional to QQQ's returns, so the true correlation is exactly 1.0.
    idx = pd.date_range("2026-01-04", periods=40, freq="W")  # Sunday-ending
    rng = np.random.default_rng(3)
    qqq_returns = rng.normal(0, 0.02, size=len(idx))
    qqq = pd.Series(300.0 * np.cumprod(1 + qqq_returns), index=idx)
    nl = pd.Series(6_000_000.0 + np.cumsum(5.0 * qqq_returns), index=idx)
    monkeypatch.setattr(MC, "net_liquidity_series", lambda years_back=3: nl)
    monkeypatch.setattr(MC, "_weekly_series_from_yahoo",
                        lambda symbol, years_back=3: qqq)
    corr, n, source = MC.net_liquidity_vs_qqq_correlation(years_back=3)
    assert corr == pytest.approx(1.0, abs=1e-6)
    assert n == len(idx) - 1  # .diff()/.pct_change() drop the first row
    assert "QQQ" in source


def test_net_liquidity_vs_qqq_correlation_none_when_either_series_empty(monkeypatch):
    monkeypatch.setattr(MC, "net_liquidity_series", lambda years_back=3: pd.Series(dtype=float))
    monkeypatch.setattr(MC, "_weekly_series_from_yahoo",
                        lambda symbol, years_back=3: pd.Series(dtype=float))
    corr, n, source = MC.net_liquidity_vs_qqq_correlation(years_back=3)
    assert (corr, n, source) == (None, None, None)


# ---------------------------------------------------------------- macro_rules.py: per-angle rules

def _policy_pack(lo, hi, implied):
    return {"policy_path": {"target_range_low": lo, "target_range_high": hi,
                            "futures_implied_rate": implied}}


def test_policy_path_supportive_when_futures_price_cuts_below_target():
    r = MR.policy_path(_policy_pack(4.0, 4.25, 3.8))  # mid 4.125, gap -0.325
    assert r["applicable"] and r["tech_impact"] == "supportive" and r["gold_impact"] == "supportive"


def test_policy_path_headwind_when_futures_price_hikes_above_target():
    r = MR.policy_path(_policy_pack(4.0, 4.25, 4.5))  # mid 4.125, gap +0.375
    assert r["tech_impact"] == "headwind" and r["gold_impact"] == "headwind"


def test_policy_path_neutral_within_gap_threshold():
    r = MR.policy_path(_policy_pack(4.0, 4.25, 4.15))  # mid 4.125, gap +0.025
    assert r["tech_impact"] == "neutral" and r["gold_impact"] == "neutral"


def test_policy_path_na_when_inputs_missing():
    r = MR.policy_path({"policy_path": {}})
    assert r["applicable"] is False and r["reason"]


def test_liquidity_supportive_when_net_liquidity_expanding():
    pack = {"liquidity": {"net_liquidity": 6_000_000.0, "net_liquidity_3m_change_pct": 5.0}}
    r = MR.liquidity(pack)
    assert r["tech_impact"] == "supportive" and r["gold_impact"] == "supportive"


def test_liquidity_headwind_when_net_liquidity_contracting():
    pack = {"liquidity": {"net_liquidity": 6_000_000.0, "net_liquidity_3m_change_pct": -4.0}}
    r = MR.liquidity(pack)
    assert r["tech_impact"] == "headwind" and r["gold_impact"] == "headwind"


def test_inflation_supportive_at_target_and_reaccelerating_flips_gold():
    at_target = MR.inflation({"inflation": {"core_pce_yoy_pct": 2.0, "core_pce_3m_annualized_pct": 2.0}})
    assert at_target["tech_impact"] == "supportive" and at_target["gold_impact"] == "neutral"
    reaccel = MR.inflation({"inflation": {"core_pce_yoy_pct": 3.0, "core_pce_3m_annualized_pct": 4.0}})
    assert reaccel["gold_impact"] == "supportive"
    assert "reaccelerating" in reaccel["label"]


def test_real_yields_headwind_when_rising_and_supportive_when_falling():
    rising = MR.real_yields({"real_yields": {"real_10y": {"value": 2.0},
                                              "real_10y_3m_change_bps": 20.0}})
    assert rising["tech_impact"] == "headwind" and rising["gold_impact"] == "headwind"
    falling = MR.real_yields({"real_yields": {"real_10y": {"value": 1.5},
                                               "real_10y_3m_change_bps": -20.0}})
    assert falling["tech_impact"] == "supportive" and falling["gold_impact"] == "supportive"


def test_growth_recession_flags_sahm_and_curve_inversion():
    pack = {"growth_recession": {"sahm_rule": {"value": 0.6},
                                 "yield_curve_10y2y": {"value": -0.3},
                                 "unemployment_rate": {"value": 4.5}}}
    r = MR.growth_recession(pack)
    assert r["tech_impact"] == "headwind" and r["gold_impact"] == "supportive"
    assert "Sahm rule triggered" in r["label"] and "yield curve inverted" in r["label"]


def test_growth_recession_no_trigger_when_below_thresholds():
    pack = {"growth_recession": {"sahm_rule": {"value": 0.1}, "yield_curve_10y2y": {"value": 0.5}}}
    r = MR.growth_recession(pack)
    assert r["tech_impact"] == "supportive" and r["gold_impact"] == "neutral"


def test_financial_conditions_stressed_vs_calm():
    stressed = MR.financial_conditions(
        {"financial_conditions": {"high_yield_spread": {"value": 7.0}, "vix": {"value": 30.0}}})
    assert stressed["tech_impact"] == "headwind" and stressed["gold_impact"] == "supportive"
    calm = MR.financial_conditions(
        {"financial_conditions": {"high_yield_spread": {"value": 3.0}, "vix": {"value": 12.0}}})
    assert calm["tech_impact"] == "supportive"


def test_dollar_fx_asymmetric_tech_neutral_gold_directional():
    strong = MR.dollar_fx({"dollar_fx": {"broad_dollar_index_3m_change_pct": 4.0,
                                         "broad_dollar_index": {"value": 105.0}}})
    assert strong["tech_impact"] == "neutral" and strong["gold_impact"] == "headwind"
    weak = MR.dollar_fx({"dollar_fx": {"broad_dollar_index_3m_change_pct": -4.0,
                                       "broad_dollar_index": {"value": 95.0}}})
    assert weak["gold_impact"] == "supportive"


def test_event_calendar_flags_binary_event_within_3_days():
    # binary_event_within_3d is precomputed by macro.build_macro_pack (see
    # its 0<=days<=3 check) - the rule only reads it, never recomputes it.
    soon = MR.event_calendar({"event_calendar": {"next_fomc_date": "2026-09-08",
                                                  "next_fomc_days": 3,
                                                  "binary_event_within_3d": True}})
    assert soon["applicable"] and "avoid new sizing" in soon["label"]
    far = MR.event_calendar({"event_calendar": {"next_fomc_date": "2026-10-15",
                                                 "next_fomc_days": 40,
                                                 "binary_event_within_3d": False}})
    assert "avoid new sizing" not in far["label"]


def test_all_angles_na_with_reason_when_pack_is_empty():
    card = MR.scorecard({})
    for name, result in card["angles"].items():
        assert result["applicable"] is False, name
        assert result["reason"]
    assert card["coverage"] == 0.0


# ---------------------------------------------------------------- macro_rules.py: scorecard/tilt

def _card_with_impacts(tech_impacts: list[str], gold_impacts: list[str]) -> dict:
    angles = {}
    for i, (t, g) in enumerate(zip(tech_impacts, gold_impacts)):
        angles[f"angle{i}"] = {"applicable": True, "tech_impact": t, "gold_impact": g,
                               "label": "x", "detail": "x"}
    n_applicable = len(angles)
    return {"angles": angles, "coverage": n_applicable / max(n_applicable, 1),
            "coverage_detail": f"{n_applicable}/{n_applicable} angles resolved"}


def test_tilt_adds_when_net_supportive_at_least_two():
    card = _card_with_impacts(["supportive"] * 4, ["neutral"] * 4)
    tilt = MR.regime_and_tilt(card)
    assert tilt["tech"]["tilt"] == "add" and tilt["tech"]["regime"] == "supportive"
    assert tilt["gold"]["tilt"] == "hold"


def test_tilt_waits_when_net_headwind_at_least_two():
    card = _card_with_impacts(["headwind"] * 4, ["neutral"] * 4)
    tilt = MR.regime_and_tilt(card)
    assert tilt["tech"]["tilt"] == "wait" and tilt["tech"]["regime"] == "headwind"


def test_tilt_holds_below_coverage_floor_regardless_of_impacts():
    # 8 canonical angles but only 3 resolved (< MIN_COVERAGE_FOR_TILT=0.5)
    # must force "hold" even though every resolved angle is supportive.
    card = {"angles": {f"a{i}": {"applicable": True, "tech_impact": "supportive",
                                 "gold_impact": "supportive", "label": "x", "detail": "x"}
                       for i in range(3)},
           "coverage": 3 / 8, "coverage_detail": "3/8 angles resolved"}
    tilt = MR.regime_and_tilt(card)
    assert tilt["tech"]["tilt"] == "hold" and tilt["tech"]["regime"] == "unmeasured"
    assert "too thin" in tilt["tech"]["reason"]


def test_scorecard_coverage_reflects_fraction_of_applicable_angles():
    # Only policy_path and liquidity have enough inputs to resolve; the
    # other six canonical angles fall through to na() - coverage is a
    # straight count of resolved angles, not a weighted/special heuristic.
    pack = {
        "policy_path": {"target_range_low": 4.0, "target_range_high": 4.25,
                        "futures_implied_rate": 4.1},
        "liquidity": {"net_liquidity": 6_000_000.0, "net_liquidity_3m_change_pct": 1.0},
    }
    card = MR.scorecard(pack)
    assert card["coverage"] == pytest.approx(2 / len(MR.ANGLES))
    assert card["coverage_detail"] == f"2/{len(MR.ANGLES)} angles resolved"


# ---------------------------------------------------------------- macro_rules.py: FOMC redline

def test_fomc_redline_na_when_no_previous_statement_to_diff_against():
    # First-ever run: only the latest statement is available, no prev yet.
    r = MR.fomc_redline({"fomc": {"latest_text": "x"}})
    assert r == {"applicable": False, "reason": "no previous FOMC statement to diff against",
                "diff_lines": []}


def test_fomc_redline_diffs_only_changed_sentences():
    pack = {"fomc": {
        "prev_text": ("The Committee decided to maintain the target range. "
                      "Economic activity has expanded at a solid pace."),
        "latest_text": ("The Committee decided to lower the target range. "
                        "Economic activity has expanded at a solid pace."),
        "prev_date": "2026-06-15", "latest_date": "2026-07-30"}}
    r = MR.fomc_redline(pack)
    assert r["applicable"] and r["changed"]
    joined = "\n".join(r["diff_lines"])
    assert "lower the target range" in joined and "maintain the target range" in joined
    # unchanged sentence must not appear in the diff at all
    assert "expanded at a solid pace" not in joined


def test_fomc_redline_no_change_when_statements_are_identical():
    text = "The Committee decided to maintain the target range."
    pack = {"fomc": {"prev_text": text, "latest_text": text,
                     "prev_date": "2026-06-15", "latest_date": "2026-07-30"}}
    r = MR.fomc_redline(pack)
    assert r["applicable"] and not r["changed"] and r["diff_lines"] == []


def test_fomc_redline_sentence_split_survives_middle_initials():
    # Regression: a naive ". "-split shredded "Beth M. Hammack, ... Lorie K.
    # Logan" into three fake sentences (see macro_rules.fomc_redline docstring).
    text = "Beth M. Hammack and Lorie K. Logan voted for this action."
    pack = {"fomc": {"prev_text": text, "latest_text": text + " ",
                     "prev_date": "d1", "latest_date": "d2"}}
    r = MR.fomc_redline(pack)
    # identical content (only whitespace differs after strip) -> no sentence
    # was spuriously split into "M." or "K." fragments that would diff.
    assert not r["changed"]


def test_fomc_redline_na_when_latest_statement_missing():
    r = MR.fomc_redline({"fomc": {}})
    assert r == {"applicable": False, "reason": "latest FOMC statement unavailable", "diff_lines": []}

"""Hard post-LLM guardrail tests: pure function, no network/LLM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import guardrails as G  # noqa: E402


def _pack(regime="mature-profitable", fair_value=150.0, coverage=0.8, data_warning=None):
    return {
        "fundamentals": {"data_warning": data_warning},
        "valuation": {
            "regime": {"regime": regime}, "fair_value": fair_value,
            "data_coverage": {"applicable": True, "value": coverage},
        },
    }


def _decision(signal="buy", conviction=70, size_pct=5.0):
    return {"signal": signal, "conviction": conviction, "size_pct": size_pct,
           "action": signal.upper(), "invalidation": "x", "rationale": "y",
           "horizon_months": 12}


def _analysts(citation_ok=True):
    return {name: {"signal": "bullish", "citations": {"citation_ok": citation_ok}}
           for name in ("fundamental", "valuation", "technical", "sentiment", "buffett", "burry")}


def test_no_override_when_thesis_and_coverage_and_citations_are_fine():
    out = G.apply_guardrails(_pack(), _analysts(), _decision(conviction=70))
    assert out["conviction"] == 70
    assert out["guardrail_notes"] == []


def test_caps_conviction_when_regime_has_no_fair_value():
    out = G.apply_guardrails(_pack(regime="etf", fair_value=None), _analysts(),
                             _decision(signal="hold", conviction=80))
    assert out["conviction"] == G.MAX_CONVICTION_NO_THESIS
    assert any("no valuation model" in n for n in out["guardrail_notes"])


def test_caps_conviction_when_regime_unknown():
    out = G.apply_guardrails(_pack(regime="unknown", fair_value=None), _analysts(),
                             _decision(signal="hold", conviction=90))
    assert out["conviction"] == G.MAX_CONVICTION_NO_THESIS


def test_does_not_cap_when_already_below_threshold():
    out = G.apply_guardrails(_pack(regime="etf", fair_value=None), _analysts(),
                             _decision(signal="hold", conviction=10, size_pct=0.0))
    assert out["conviction"] == 10
    assert out["guardrail_notes"] == []


def test_caps_conviction_on_low_data_coverage():
    out = G.apply_guardrails(_pack(coverage=0.07), _analysts(), _decision(conviction=80))
    assert out["conviction"] == G.MAX_CONVICTION_LOW_COVERAGE
    assert any("data coverage 7%" in n for n in out["guardrail_notes"])


def test_notes_data_warning_without_capping_conviction():
    out = G.apply_guardrails(_pack(data_warning="price*shares vs market_cap mismatch"),
                             _analysts(), _decision(conviction=70))
    assert out["conviction"] == 70  # data_warning alone doesn't cap - it's flagged, not fatal
    assert any("data warning present" in n for n in out["guardrail_notes"])


def test_caps_conviction_when_majority_of_seats_fail_citation_check():
    analysts = _analysts(citation_ok=True)
    for name in ("fundamental", "buffett", "burry", "technical"):  # 4 of 6 seats
        analysts[name]["citations"]["citation_ok"] = False
    out = G.apply_guardrails(_pack(), analysts, _decision(conviction=90))
    assert out["conviction"] == G.MAX_CONVICTION_LOW_COVERAGE
    assert any("citation check failed" in n for n in out["guardrail_notes"])
    assert any("majority of seats" in n for n in out["guardrail_notes"])


def test_notes_but_does_not_cap_when_minority_of_seats_fail_citation_check():
    analysts = _analysts(citation_ok=True)
    analysts["burry"]["citations"]["citation_ok"] = False  # 1 of 6 seats
    out = G.apply_guardrails(_pack(), analysts, _decision(conviction=90))
    assert out["conviction"] == 90
    assert any("citation check failed for: burry" in n for n in out["guardrail_notes"])
    assert not any("majority" in n for n in out["guardrail_notes"])


def test_forces_size_to_zero_on_non_buy_even_if_decision_disagreed():
    # defense-in-depth: run_decision already enforces this at the source,
    # but the guardrail must not silently trust a nonzero size either.
    out = G.apply_guardrails(_pack(), _analysts(), _decision(signal="sell", size_pct=8.0))
    assert out["size_pct"] == 0.0
    assert any("size forced 0" in n for n in out["guardrail_notes"])


def test_leaves_buy_size_untouched():
    out = G.apply_guardrails(_pack(), _analysts(), _decision(signal="buy", size_pct=4.0))
    assert out["size_pct"] == 4.0


def test_does_not_mutate_input_decision_dict():
    decision = _decision(conviction=90)
    original = dict(decision)
    G.apply_guardrails(_pack(regime="etf", fair_value=None), _analysts(), decision)
    assert decision == original


def test_judge_disagreement_caps_conviction_and_is_noted():
    d = _decision("sell", 80, 0.0)
    d["judge_agreement"] = {"signals": ["sell", "buy"], "agree": False}
    out = G.apply_guardrails(_pack(), {}, d)
    assert out["conviction"] == G.MAX_CONVICTION_UNSTABLE
    assert any("disagreed" in n for n in out["guardrail_notes"])


def test_ablation_flip_says_decision_rests_on_valuation_and_caps():
    d = _decision("sell", 80, 0.0)
    d["ablation"] = {"signal": "buy", "conviction": 60, "flipped": True}
    out = G.apply_guardrails(_pack(), {}, d)
    assert out["conviction"] == G.MAX_CONVICTION_UNSTABLE
    assert any("rests on the valuation model" in n for n in out["guardrail_notes"])


def test_agreeing_judges_and_unflipped_ablation_change_nothing():
    d = _decision("sell", 80, 0.0)
    d["judge_agreement"] = {"signals": ["sell", "sell"], "agree": True}
    d["ablation"] = {"signal": "sell", "conviction": 70, "flipped": False}
    out = G.apply_guardrails(_pack(), {}, d)
    assert out["conviction"] == 80 and out["guardrail_notes"] == []

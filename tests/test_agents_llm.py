"""Tests for fetch_market_consensus: the web-search fallback for analyst
growth/target data when quoteSummary is blocked (see data.py for the sibling
fast_info/SEC/beta fallbacks). Only the pure parsing/validation logic is
tested here - Agent.prompt itself is monkeypatched, no live network/API call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

import agents_llm as A


@dataclass
class _FakeResult:
    status: str
    result: str | None


def _client() -> A.LLMClient:
    return A.LLMClient(api_key="test-key", model="test-model")  # type: ignore[arg-type]


def _patch_prompt(monkeypatch, result: _FakeResult):
    monkeypatch.setattr(A.Agent, "prompt", staticmethod(lambda prompt, options: result))


def _budget() -> A.Budget:
    return A.Budget()


def test_parse_signal_extracts_cited_numbers():
    reply = ('{"signal": "bullish", "confidence": 70, "reasoning": "ok", '
             '"cited_numbers": [48.16, 315.18, "not a number", 12]}')
    out = A.parse_signal(reply)
    assert out["cited_numbers"] == [48.16, 315.18, 12.0]


def test_parse_signal_fallback_has_empty_cited_numbers():
    out = A.parse_signal("no json here, just bullish prose")
    assert out["cited_numbers"] == []


def test_validate_citations_verifies_numbers_present_in_pack():
    pack_json = json.dumps({"valuation": {"fair_value": 48.16}, "price": 315.18})
    out = A.validate_citations([48.16, 315.18], pack_json)
    assert out == {"total": 2, "verified": 2, "unverified": [], "citation_ok": True}


def test_validate_citations_flags_a_number_not_in_the_pack():
    # regression target: a devil's-advocate seat was once seen doing inline
    # CAPM math and citing the result as if it were a pack number.
    pack_json = json.dumps({"valuation": {"fair_value": 48.16}})
    out = A.validate_citations([48.16, 9.87], pack_json)
    assert out["total"] == 2
    assert out["verified"] == 1
    assert out["unverified"] == [9.87]
    assert out["citation_ok"] is False


def test_validate_citations_tolerates_rounding_in_restated_numbers():
    pack_json = json.dumps({"fair_value": 48.16})
    out = A.validate_citations([48.2], pack_json)  # rounded when restated in prose
    assert out["citation_ok"] is True


def test_validate_citations_no_claims_is_ok_not_penalized():
    # an empty cited_numbers list (e.g. the no-domain-evidence fallback) is
    # not the same failure mode as a claim that fails to verify.
    out = A.validate_citations([], json.dumps({"price": 1.0}))
    assert out["citation_ok"] is True
    assert out["total"] == 0


def test_validate_citations_survives_malformed_pack_json():
    out = A.validate_citations([1.0], "not json")
    assert out["citation_ok"] is False
    assert out["unverified"] == [1.0]


@pytest.mark.parametrize("signal,judge_size,expected_size,expected_action", [
    ("sell", 8.0, 0.0, "AVOID"),   # long-only: no sizing a sell/short
    ("hold", 3.0, 0.0, "HOLD"),    # long-only: no sizing a hold either
    ("buy", 4.0, 4.0, "BUY"),      # buy is the only signal that keeps its size
])
def test_run_decision_enforces_long_only_size_and_action_label(
        monkeypatch, signal, judge_size, expected_size, expected_action):
    reply = (f'Some reasoning. {{"signal": "{signal}", "conviction": 70, '
             f'"size_pct": {judge_size}, "horizon_months": 12, '
             f'"invalidation": "x", "rationale": "y"}}')
    monkeypatch.setattr(A, "ask", lambda client, system, user: reply)
    out = A.run_decision(_client(), "{}", {}, _budget())
    d = out["decision"]
    assert d["signal"] == signal
    assert d["size_pct"] == expected_size
    assert d["action"] == expected_action


def test_run_decision_horizon_comes_from_the_user_not_the_judge_json(monkeypatch):
    prompts = []
    reply = ('{"signal": "buy", "conviction": 60, "size_pct": 2, "horizon_months": 3, '
             '"invalidation": "x", "rationale": "y"}')

    def fake_ask(client, system, user):
        prompts.append(system)
        return reply

    monkeypatch.setattr(A, "ask", fake_ask)
    out = A.run_decision(_client(), "{}", {}, _budget(), horizon_months=6)
    assert out["decision"]["horizon_months"] == 6
    trader_prompt, judge_prompt = prompts[0], prompts[-1]
    assert "6 months" in trader_prompt and "6 months" in judge_prompt
    assert '"horizon_months"' not in judge_prompt  # judge no longer asked for it


def test_run_decision_defaults_to_twelve_months(monkeypatch):
    monkeypatch.setattr(A, "ask", lambda client, system, user: "no json")
    out = A.run_decision(_client(), "{}", {}, _budget())
    assert out["decision"]["horizon_months"] == A.DEFAULT_HORIZON_MONTHS == 12


def test_run_analysts_fills_the_horizon_into_persona_prompts(monkeypatch):
    seen = {}

    def fake_ask(client, system, user):
        seen[system.split(".")[0]] = system
        return '{"signal": "neutral", "confidence": 50, "reasoning": "r", "cited_numbers": []}'

    monkeypatch.setattr(A, "ask", fake_ask)
    A.run_analysts(_client(), {"horizon_months": 3, "fundamentals": {}}, "{}", "{}", A.Budget())
    technical = next(s for s in seen.values() if s.startswith("You are a technical analyst"))
    macro = next(s for s in seen.values() if s.startswith("You are a macro strategist"))
    assert "3-month-forward hit rate" in technical
    assert "3-month holding period" in macro
    assert all("{horizon_months}" not in s for s in seen.values())


def test_run_decision_fallback_decision_is_long_only_safe(monkeypatch):
    monkeypatch.setattr(A, "ask", lambda client, system, user: "no json here at all")
    out = A.run_decision(_client(), "{}", {}, _budget())
    assert out["decision"]["size_pct"] == 0.0
    assert out["decision"]["action"] == "HOLD"


def test_fetch_market_consensus_parses_valid_reply_with_sources(monkeypatch):
    reply = ('{"growth_est_5y_pct": 12.5, "analyst_target_mean": 210.0, '
             '"analyst_target_median": 205.0, "analyst_count": 34, '
             '"as_of": "2026-08-15", "sources": ["https://finance.yahoo.com/quote/AAPL"]}')
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))

    out = A.fetch_market_consensus(_client(), "AAPL", "Apple Inc.", A.Budget())

    assert out["growth_est"] == pytest.approx(0.125)  # percent -> fraction
    assert out["analyst_target_mean"] == 210.0
    assert out["analyst_target_median"] == 205.0
    assert out["analyst_target_count"] == 34
    assert out["consensus_sources"] == ["https://finance.yahoo.com/quote/AAPL"]
    assert out["consensus_as_of"] == "2026-08-15"


def test_fetch_market_consensus_rejects_reply_with_no_sources(monkeypatch):
    # Even a well-formed, plausible-looking number is unusable without a
    # citation - this is the guardrail against the model inventing figures.
    reply = '{"growth_est_5y_pct": 12.5, "analyst_target_mean": 210.0, "sources": []}'
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))

    assert A.fetch_market_consensus(_client(), "AAPL", None, A.Budget()) == {}


def test_fetch_market_consensus_rejects_non_url_sources(monkeypatch):
    reply = '{"growth_est_5y_pct": 12.5, "sources": ["trust me bro"]}'
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))

    assert A.fetch_market_consensus(_client(), "AAPL", None, A.Budget()) == {}


def test_fetch_market_consensus_handles_unparseable_reply(monkeypatch):
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result="not json at all"))
    assert A.fetch_market_consensus(_client(), "AAPL", None, A.Budget()) == {}


def test_fetch_market_consensus_handles_run_error_status(monkeypatch):
    _patch_prompt(monkeypatch, _FakeResult(status="error", result=None))
    assert A.fetch_market_consensus(_client(), "AAPL", None, A.Budget()) == {}


def test_fetch_market_consensus_handles_prompt_exception(monkeypatch):
    def boom(prompt, options):
        raise RuntimeError("network down")

    monkeypatch.setattr(A.Agent, "prompt", staticmethod(boom))
    assert A.fetch_market_consensus(_client(), "AAPL", None, A.Budget()) == {}


def test_fetch_market_consensus_parses_recent_analyst_calls(monkeypatch):
    reply = ('{"growth_est_5y_pct": 12.5, "sources": ["https://finviz.com/x"], '
             '"recent_calls": ['
             '{"firm": "Goldman Sachs", "analyst": "Jane Doe", "rating": "Buy", '
             '"action": "Reiterated", "price_target": 360.0, "date": "2026-07-31"}, '
             '{"firm": "Barclays", "analyst": null, "rating": "Underweight", '
             '"action": "Reiterated", "price_target": 245.0, "date": "2026-07-31"}'
             ']}')
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))

    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())

    calls = out["recent_analyst_calls"]
    assert len(calls) == 2
    assert calls[0] == {"firm": "Goldman Sachs", "analyst": "Jane Doe", "rating": "Buy",
                        "action": "Reiterated", "price_target": 360.0, "date": "2026-07-31"}
    assert calls[1]["analyst"] is None
    assert calls[1]["firm"] == "Barclays"


def test_fetch_market_consensus_drops_calls_missing_a_named_firm(monkeypatch):
    # firm is the one non-negotiable field - an entry with no real named firm
    # is exactly the kind of invented-looking data the guardrail exists for.
    reply = ('{"sources": ["https://finviz.com/x"], "recent_calls": ['
             '{"firm": "", "rating": "Buy"}, '
             '{"analyst": "Someone", "rating": "Buy"}, '
             '{"firm": "Real Firm LLC", "rating": "Hold"}'
             ']}')
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))

    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())
    assert len(out["recent_analyst_calls"]) == 1
    assert out["recent_analyst_calls"][0]["firm"] == "Real Firm LLC"


def test_fetch_market_consensus_caps_recent_calls_at_five(monkeypatch):
    calls_json = ", ".join(f'{{"firm": "Firm {i}"}}' for i in range(10))
    reply = f'{{"sources": ["https://finviz.com/x"], "recent_calls": [{calls_json}]}}'
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))

    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())
    assert len(out["recent_analyst_calls"]) == 5


def test_has_statement_evidence_true_when_any_core_field_numeric():
    assert A.has_statement_evidence({"fundamentals": {"revenue": 1000.0}})
    assert A.has_statement_evidence({"fundamentals": {"ebit": 0.0}})  # 0 is a real value, not missing
    assert not A.has_statement_evidence({"fundamentals": {"revenue": None, "price": 89.37}})
    assert not A.has_statement_evidence({"fundamentals": {}})


def test_run_analysts_gates_statement_dependent_seats_without_a_model_call(monkeypatch):
    """ETF/blocked-quoteSummary pack: fundamental/buffett/burry must not call
    the model at all (the fake-consensus bug), while technical/sentiment/
    valuation still do."""
    calls: list[str] = []

    def fake_ask(client, system, user):
        calls.append(system)
        return '{"signal": "bullish", "confidence": 60, "reasoning": "tape only"}'

    monkeypatch.setattr(A, "ask", fake_ask)
    pack = {"fundamentals": {"quote_type": "ETF", "price": 89.37}}
    out = A.run_analysts(_client(), pack, "{}", "{}", A.Budget())

    for seat in A.STATEMENT_DEPENDENT_SEATS:
        assert out[seat]["signal"] == "neutral"
        assert out[seat]["confidence"] == 0
        assert out[seat]["reasoning"] == A.NO_DOMAIN_EVIDENCE_REASON
        assert out[seat]["citations"]["citation_ok"] is True
    called_seats = len(calls)
    assert called_seats == len(A.ANALYSTS) - len(A.STATEMENT_DEPENDENT_SEATS)


def test_run_analysts_calls_every_seat_when_statements_present(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(A, "ask", lambda client, system, user: (
        calls.append(system),
        '{"signal": "bullish", "confidence": 60, "reasoning": "ok"}')[1])
    pack = {"fundamentals": {"revenue": 1000.0}}
    A.run_analysts(_client(), pack, "{}", "{}", A.Budget())
    assert len(calls) == len(A.ANALYSTS)


def test_run_analysts_routes_redacted_pack_to_statement_dependent_seats(monkeypatch):
    """Even with statements present (no ETF gating), fundamental/buffett/
    burry must read the redacted JSON, not the full one - the fair-value
    leak happened on full-coverage names like AAPL, not only on ETFs."""
    seen: dict[str, str] = {}
    monkeypatch.setattr(A, "ask", lambda client, system, user: (
        seen.__setitem__(system, user),
        '{"signal": "neutral", "confidence": 50, "reasoning": "ok"}')[1])
    pack = {"fundamentals": {"revenue": 1000.0}}
    A.run_analysts(_client(), pack, "FULL_PACK_MARKER", "REDACTED_PACK_MARKER", A.Budget())
    for name, persona in A.ANALYSTS.items():
        filled = persona.replace("{horizon_months}", str(A.DEFAULT_HORIZON_MONTHS))
        user = seen[filled + " " + A.SCHEMA]
        expected_marker = "REDACTED_PACK_MARKER" if name in A.STATEMENT_DEPENDENT_SEATS else "FULL_PACK_MARKER"
        assert expected_marker in user, f"{name} seat got the wrong pack view"


def test_fetch_market_consensus_ignores_non_numeric_fields(monkeypatch):
    reply = ('{"growth_est_5y_pct": "n/a", "analyst_target_mean": null, '
             '"sources": ["https://example.com/x"]}')
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))

    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())
    assert "growth_est" not in out
    assert "analyst_target_mean" not in out
    assert out["consensus_sources"] == ["https://example.com/x"]


def test_fetch_market_consensus_parses_eps_fwd(monkeypatch):
    # unlocks the AEG model (the only non-fade point model) when
    # quoteSummary's forwardEps is blocked - see pipeline.build_pack.
    reply = '{"eps_fwd_next_fy": 7.85, "sources": ["https://finviz.com/x"]}'
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))
    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())
    assert out["eps_fwd"] == 7.85


def test_fetch_market_consensus_rejects_non_positive_eps_fwd(monkeypatch):
    reply = '{"eps_fwd_next_fy": -1.2, "sources": ["https://finviz.com/x"]}'
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))
    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())
    assert "eps_fwd" not in out


def test_fetch_market_consensus_parses_eps_revision_trend(monkeypatch):
    reply = ('{"eps_revision_90d": "up", "eps_revision_90d_detail": '
             '"12 estimates raised, 2 lowered in the last 90 days", '
             '"sources": ["https://finviz.com/x"]}')
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))
    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())
    assert out["eps_revision_90d"] == "up"
    assert out["eps_revision_90d_detail"] == "12 estimates raised, 2 lowered in the last 90 days"


def test_fetch_market_consensus_normalizes_eps_revision_trend_case(monkeypatch):
    reply = '{"eps_revision_90d": "  Down ", "sources": ["https://finviz.com/x"]}'
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))
    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())
    assert out["eps_revision_90d"] == "down"


def test_fetch_market_consensus_rejects_invalid_eps_revision_trend_value(monkeypatch):
    reply = '{"eps_revision_90d": "bullish", "sources": ["https://finviz.com/x"]}'
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))
    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())
    assert "eps_revision_90d" not in out


def test_fetch_market_consensus_omits_eps_revision_trend_when_absent(monkeypatch):
    reply = '{"growth_est_5y_pct": 12.5, "sources": ["https://finviz.com/x"]}'
    _patch_prompt(monkeypatch, _FakeResult(status="completed", result=reply))
    out = A.fetch_market_consensus(_client(), "AAPL", None, A.Budget())
    assert "eps_revision_90d" not in out
    assert "eps_revision_90d_detail" not in out


@pytest.mark.parametrize("signals,first,unanimous", [
    (["bearish", "bearish", "bullish"], "## Bull case", False),   # against a bearish majority
    (["bullish", "bullish", "bearish"], "## Bear case", False),   # against a bullish majority
    (["bullish", "bearish"], "## Bull case", False),              # tie counts as bearish
    (["neutral", "neutral"], "## Bull case", True),
])
def test_trader_leads_with_the_case_against_the_majority(signals, first, unanimous):
    analysts = {f"s{i}": {"signal": s} for i, s in enumerate(signals)}
    prompt = A.trader_system(analysts, 6)
    other = "## Bear case" if first == "## Bull case" else "## Bull case"
    assert prompt.index(first) < prompt.index(other) < prompt.index("## Weighing")
    assert "6-month holding period" in prompt and "## Plan" in prompt
    assert ("unanimously" in prompt) is unanimous


def test_run_decision_has_no_debate_and_shares_votes(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_ask(client, system, user):
        calls.append((system, user))
        return "RISK TURN" if "risk manager" in system else '{"signal": "hold"}'

    monkeypatch.setattr(A, "ask", fake_ask)
    analysts = {"technical": {"signal": "bullish", "reasoning": "VOTE-MARKER"}}
    A.run_decision(_client(), "{}", analysts, _budget(), pack_json_redacted="{}")
    assert len(calls) == 1 + 3 + 3  # trader, 3 risk seats, 2 judges + ablation
    assert calls[0][0].startswith("You are the trader")
    risk = [u for s, u in calls if "risk manager" in s]
    judges = [u for s, u in calls if s.startswith("You are the Risk Judge")]
    assert all("VOTE-MARKER" in u for u in risk + judges)
    # neutral (last) seat sees both earlier turns in full
    assert risk[-1].count("RISK TURN") == 2


def test_run_decision_caps_risk_seats_and_truncates_overlong_turns(monkeypatch):
    calls: list[tuple[str, str]] = []
    huge = "x" * (A.TURN_CHAR_CAP * 10)

    def fake_ask(client, system, user):
        calls.append((system, user))
        return huge if "risk manager" in system else '{"signal": "hold"}'

    monkeypatch.setattr(A, "ask", fake_ask)
    out = A.run_decision(_client(), "{}", {}, _budget())
    risk_systems = [s for s, _ in calls if "risk manager" in s]
    assert len(risk_systems) == 3
    assert all(A.TURN_RULE in s for s in risk_systems)
    assert all(len(t) < A.TURN_CHAR_CAP + 100 for t in out["risk_transcript"])
    judge_user = calls[-1][1]
    assert len(judge_user) < 5 * (A.TURN_CHAR_CAP + 100)


def test_capped_leaves_short_text_and_marks_truncation():
    assert A._capped("short") == "short"
    long = A._capped("y" * (A.TURN_CHAR_CAP + 1))
    assert long.startswith("y" * A.TURN_CHAR_CAP) and long.endswith("[truncated: over the turn limit]")


def test_resolve_model_matches_variant_params_not_display_names(monkeypatch):
    from types import SimpleNamespace as NS

    def variant(name, default, **params):
        return NS(display_name=name, is_default=default,
                  params=[NS(id=k, value=v) for k, v in params.items()])

    catalog = [NS(id="grok-4.7", variants=[
        variant("Grok 4.7  High Fast\u200b", True, context="500k", reasoning_effort="high", fast="true"),
        variant("Grok 4.7  Medium\u200b", False, context="256k", reasoning_effort="medium"),
        variant("Grok 4.7  Medium Fast\u200b", False, context="500k", reasoning_effort="medium", fast="true"),
        variant("Grok 4.7  Medium Fast\u200b", False, context="256k", reasoning_effort="medium", fast="true"),
    ])]
    monkeypatch.setattr(A.Cursor.models, "list", lambda api_key: catalog)
    sel = A.resolve_model("k")
    assert sel.id == "grok-4.7"
    assert {p.id: p.value for p in sel.params} == A.MODEL_VARIANT_PARAMS


def test_ask_timeout_does_not_wait_for_the_hung_call(monkeypatch):
    import time

    monkeypatch.setattr(A, "CALL_TIMEOUT", 0.1)
    monkeypatch.setattr(A.Agent, "prompt", staticmethod(lambda prompt, options: time.sleep(1.5)))
    start = time.monotonic()
    with pytest.raises(A.PipelineError, match="timed out"):
        A.ask(_client(), "sys", "user")
    assert time.monotonic() - start < 1.0


def test_ask_timeout_does_not_wait_for_the_hung_call(monkeypatch):
    import time

    monkeypatch.setattr(A, "CALL_TIMEOUT", 0.1)
    monkeypatch.setattr(A.Agent, "prompt", staticmethod(lambda prompt, options: time.sleep(1.5)))
    start = time.monotonic()
    with pytest.raises(A.PipelineError, match="timed out"):
        A.ask(_client(), "sys", "user")
    assert time.monotonic() - start < 1.0


def test_ask_runs_outside_the_workspace_with_a_no_tools_preamble(monkeypatch):
    seen = {}

    def fake_prompt(prompt, options):
        seen["prompt"], seen["options"] = prompt, options
        return _FakeResult(status="finished", result="ok")

    monkeypatch.setattr(A.Agent, "prompt", staticmethod(fake_prompt))
    A.ask(_client(), "sys", "user")
    assert seen["prompt"].startswith(A.TEXT_ONLY_PREAMBLE)
    assert seen["options"].local.cwd == str(A.AGENT_CWD)
    assert not str(A.AGENT_CWD).startswith(str(A.BASE))


def _judge_json(signal, conviction, size=0, price=None, direction=None):
    extra = f', "invalidation_price": {price}, "invalidation_direction": "{direction}"' if price else ""
    return (f'{{"signal": "{signal}", "conviction": {conviction}, "size_pct": {size}, '
            f'"invalidation": "x"{extra}, "rationale": "r"}}')


def _run_with_judges(monkeypatch, judge_replies, ablation_reply=None):
    replies = iter(judge_replies)
    seen = {"ablation_user": None}

    def fake_ask(client, system, user):
        if "independent Risk Judge" in system:
            seen["ablation_user"] = user
            return ablation_reply
        if "Risk Judge" in system:
            return next(replies)
        return "text"

    monkeypatch.setattr(A, "ask", fake_ask)
    out = A.run_decision(_client(), "{}", {"valuation": {"signal": "bearish"}, "macro": {"signal": "neutral"}},
                         _budget(), 12,
                         pack_json_redacted="REDACTED" if ablation_reply else None)
    return out, seen


def test_two_judges_agreeing_keep_lower_conviction_and_smaller_size(monkeypatch):
    out, _ = _run_with_judges(monkeypatch, [_judge_json("buy", 70, 6), _judge_json("buy", 60, 4)])
    d = out["decision"]
    assert (d["signal"], d["conviction"], d["size_pct"]) == ("buy", 60, 4.0)
    assert d["judge_agreement"] == {"signals": ["buy", "buy"], "agree": True}


def test_two_judges_disagreeing_take_the_more_cautious_signal(monkeypatch):
    out, _ = _run_with_judges(monkeypatch, [_judge_json("buy", 80, 8), _judge_json("hold", 65)])
    d = out["decision"]
    assert (d["signal"], d["size_pct"], d["conviction"]) == ("hold", 0.0, 65)
    assert d["judge_agreement"]["agree"] is False


def test_one_unparseable_judge_falls_back_to_the_other(monkeypatch):
    out, _ = _run_with_judges(monkeypatch, ["no json", _judge_json("sell", 70)])
    d = out["decision"]
    assert d["signal"] == "sell" and d["judge_agreement"] is None


def test_ablation_sees_redacted_pack_without_valuation_vote_and_flags_flip(monkeypatch):
    out, seen = _run_with_judges(monkeypatch, [_judge_json("sell", 70)] * 2,
                                 ablation_reply=_judge_json("buy", 55, 3))
    assert out["decision"]["ablation"] == {"signal": "buy", "conviction": 55, "flipped": True}
    assert "REDACTED" in seen["ablation_user"]
    assert '"valuation"' not in seen["ablation_user"] and '"macro"' in seen["ablation_user"]


def test_invalidation_level_parsed_only_when_price_and_direction_valid(monkeypatch):
    out, _ = _run_with_judges(monkeypatch, [_judge_json("sell", 70, price=294, direction="below")] * 2)
    assert out["decision"]["invalidation_level"] == {"price": 294.0, "direction": "below"}
    out, _ = _run_with_judges(monkeypatch, [_judge_json("sell", 70, price=294, direction="sideways")] * 2)
    assert out["decision"]["invalidation_level"] is None


# --- provider selection (LLM_PROVIDER) ---------------------------------------

def test_openai_provider_needs_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(A.PipelineError, match="LLM_MODEL"):
        A.llm_client()


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(A.PipelineError, match="unknown LLM_PROVIDER"):
        A.llm_client()


def test_openai_provider_client_and_ask(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = A.llm_client()
    assert (client.provider, client.model.id, client.base_url) == (
        "openai", "some-model", "http://localhost:11434/v1")
    seen = {}
    monkeypatch.setattr(A, "_openai_complete",
                        lambda c, prompt: seen.setdefault("prompt", prompt) and "hello")
    assert A.ask(client, "sys", "usr") == "hello"
    assert "sys" in seen["prompt"] and "usr" in seen["prompt"]


def test_openai_provider_skips_web_search(monkeypatch):
    client = A.LLMClient(api_key="k", model=A.OpenAIModel("m"), provider="openai")
    monkeypatch.setattr(A.Agent, "prompt", staticmethod(
        lambda prompt, options: pytest.fail("web search must not call the Cursor SDK")))
    assert A._web_search_agent(client, "find consensus") == ""

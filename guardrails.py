"""Hard post-LLM guardrails: mechanical checks applied AFTER the risk judge
issues a decision, capable of overriding it.

The committee's discipline (long-only, "unknown regime is a no-view", "cite
pack numbers") is enforced today mostly through prompt instructions - a
devil's-advocate seat was once seen doing inline CAPM math despite being
told not to, and reports have shipped with 80/100 conviction on a regime
with zero statement evidence behind it. A prompt is a request, not
enforcement. This module is the enforcement layer: it never asks the model
anything, it only recomputes conviction/size/signal from facts already in
the pack and the analyst votes, matching this codebase's "LLM never does
math" posture (guardrails compute, the LLM only explains).
"""

from __future__ import annotations

# No firm-level thesis is possible without an applicable valuation model
# (unknown/etf regime, or a regime that resolved but produced no fair
# value) - a high-conviction directional call here is really a technicals/
# macro/sentiment call dressed up as fundamentals confidence.
MAX_CONVICTION_NO_THESIS = 30
# Below this data-coverage fraction, the statements behind any model are too
# thin to support a strong call even when a regime/model did resolve.
LOW_COVERAGE_THRESHOLD = 0.30
MAX_CONVICTION_LOW_COVERAGE = 40
# If citations fail for more than half the seats, the committee's numeric
# grounding as a whole is suspect, not just one seat's.
MAX_CITATION_FAILURE_RATE = 0.5
# Two judge runs on identical input disagreeing, or the call changing once the
# model-derived fair value is removed, means the call is not robust.
MAX_CONVICTION_UNSTABLE = 50


def apply_guardrails(pack: dict, analysts: dict, decision: dict) -> dict:
    """Returns a NEW decision dict (never mutates the input) with
    conviction/size mechanically capped when it disagrees with the judge,
    plus a `guardrail_notes` list naming every override so the report shows
    what changed and why - a silent override would just move the
    groundedness problem rather than fix it."""
    d = dict(decision)
    notes: list[str] = []
    v = pack.get("valuation", {})

    regime = v.get("regime", {}).get("regime")
    no_thesis = regime in ("unknown", "etf") or v.get("fair_value") is None
    if no_thesis and d["conviction"] > MAX_CONVICTION_NO_THESIS:
        notes.append(f"conviction capped {d['conviction']}->{MAX_CONVICTION_NO_THESIS}: "
                    f"regime={regime}, no valuation model produced a fair value")
        d["conviction"] = MAX_CONVICTION_NO_THESIS

    coverage = v.get("data_coverage", {})
    if (coverage.get("applicable") and coverage["value"] < LOW_COVERAGE_THRESHOLD
            and d["conviction"] > MAX_CONVICTION_LOW_COVERAGE):
        notes.append(f"conviction capped {d['conviction']}->{MAX_CONVICTION_LOW_COVERAGE}: "
                    f"data coverage {coverage['value'] * 100:.0f}% < "
                    f"{LOW_COVERAGE_THRESHOLD * 100:.0f}%")
        d["conviction"] = MAX_CONVICTION_LOW_COVERAGE

    if pack.get("fundamentals", {}).get("data_warning"):
        notes.append(f"data warning present: {pack['fundamentals']['data_warning']}")

    failed_seats = [name for name, a in analysts.items()
                    if not a.get("citations", {}).get("citation_ok", True)]
    if failed_seats:
        notes.append(f"citation check failed for: {', '.join(failed_seats)} "
                    "(cited a number not found in the pack they were shown)")
        failure_rate = len(failed_seats) / max(len(analysts), 1)
        if failure_rate > MAX_CITATION_FAILURE_RATE and d["conviction"] > MAX_CONVICTION_LOW_COVERAGE:
            notes.append(f"conviction additionally capped {d['conviction']}->"
                        f"{MAX_CONVICTION_LOW_COVERAGE}: majority of seats had unverifiable citations")
            d["conviction"] = MAX_CONVICTION_LOW_COVERAGE

    ja = d.get("judge_agreement")
    if ja and ja.get("agree") is False:
        notes.append(f"two risk-judge runs disagreed ({' vs '.join(ja['signals'])}): "
                    "took the more cautious signal")
        if d["conviction"] > MAX_CONVICTION_UNSTABLE:
            notes.append(f"conviction capped {d['conviction']}->{MAX_CONVICTION_UNSTABLE}: judge disagreement")
            d["conviction"] = MAX_CONVICTION_UNSTABLE
    ab = d.get("ablation")
    if ab and ab.get("flipped"):
        notes.append(f"decision rests on the valuation model: without the dollar fair value "
                    f"the judge said {ab['signal'].upper()}")
        if d["conviction"] > MAX_CONVICTION_UNSTABLE:
            notes.append(f"conviction capped {d['conviction']}->{MAX_CONVICTION_UNSTABLE}: valuation-dependent call")
            d["conviction"] = MAX_CONVICTION_UNSTABLE

    # Long-only invariant, re-asserted here as defense-in-depth. run_decision
    # already enforces this at the source; a guardrail silently trusting a
    # nonzero size on a non-buy signal would be worse than redundant.
    if d["signal"] != "buy" and d.get("size_pct", 0.0) != 0.0:
        notes.append(f"size forced 0 (long-only product, signal={d['signal']})")
        d["size_pct"] = 0.0

    d["guardrail_notes"] = notes
    return d

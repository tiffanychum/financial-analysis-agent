"""Deterministic macro rules: turn macro.build_macro_pack()'s raw numbers
into per-angle labels, a tech-ETF/gold impact scorecard, and an overall
regime + tilt. No LLM, no arithmetic surprises — every threshold is a named
constant so a report line traces back to a rule, not a vibe. Mirrors
valuation.py's contract: a rule that cannot resolve excludes itself with a
named reason (`na()`), it never guesses.

Honest limit (stated once, believed everywhere downstream): macro timing is
weak at a 12-month horizon. The tilt this module prints is for avoiding bad
entries and sizing (add/hold/wait), not calling bottoms or tops.
"""

from __future__ import annotations

import difflib
import re


def na(reason: str) -> dict:
    return {"applicable": False, "reason": reason, "label": None,
            "tech_impact": "neutral", "gold_impact": "neutral", "detail": None}


def _ok(*vals) -> bool:
    return all(v is not None for v in vals)


# ---------------------------------------------------------------- per-angle rules

# +/-15bps: roughly one "quarter of a quarter-point" of futures noise around
# a target-range midpoint - smaller gaps are noise, not a priced view.
POLICY_GAP_THRESHOLD_PCT = 0.15


def policy_path(pack: dict) -> dict:
    """Priced (fed funds futures) vs said (current target range midpoint) -
    the tradeable half of Fed-watching is the GAP between the two, not
    either number alone."""
    pp = pack.get("policy_path", {})
    lo, hi, implied = pp.get("target_range_low"), pp.get("target_range_high"), pp.get("futures_implied_rate")
    if not _ok(lo, hi, implied):
        return na("target range or futures-implied rate unavailable")
    mid = (lo + hi) / 2
    gap = implied - mid
    if gap < -POLICY_GAP_THRESHOLD_PCT:
        label, tech, gold = "market pricing rate cuts beyond the current target", "supportive", "supportive"
    elif gap > POLICY_GAP_THRESHOLD_PCT:
        label, tech, gold = "market pricing rate hikes beyond the current target", "headwind", "headwind"
    else:
        label, tech, gold = "market aligned with the current target range", "neutral", "neutral"
    return {"applicable": True, "label": label, "tech_impact": tech, "gold_impact": gold,
            "detail": f"target {lo:.2f}-{hi:.2f}% (mid {mid:.2f}%), futures-implied {implied:.2f}% "
                      f"(gap {gap:+.2f}pp)"}


# Net liquidity 3m change: +/-3% is a Bloomberg/Fed-watcher rule-of-thumb
# band for "meaningfully expanding/contracting" vs noise on a ~$5-6T base.
LIQUIDITY_CHANGE_THRESHOLD_PCT = 3.0


def liquidity(pack: dict) -> dict:
    liq = pack.get("liquidity", {})
    chg = liq.get("net_liquidity_3m_change_pct")
    level = liq.get("net_liquidity")
    if not _ok(chg, level):
        return na("net liquidity unavailable (WALCL/TGA/RRP fetch failed)")
    if chg > LIQUIDITY_CHANGE_THRESHOLD_PCT:
        label, tech, gold = "net liquidity expanding", "supportive", "supportive"
    elif chg < -LIQUIDITY_CHANGE_THRESHOLD_PCT:
        label, tech, gold = "net liquidity contracting", "headwind", "headwind"
    else:
        label, tech, gold = "net liquidity roughly flat", "neutral", "neutral"
    return {"applicable": True, "label": label, "tech_impact": tech, "gold_impact": gold,
            "detail": f"net liquidity ${level / 1e6:.2f}T, {chg:+.1f}% over 3mo "
                      "(WALCL - TGA - RRP)"}


# Fed's own target is core PCE = 2.0% YoY; +/-0.3pp is the commonly-cited
# "close enough to call it at target" band around FOMC communications.
INFLATION_TARGET_PCT = 2.0
INFLATION_AT_TARGET_BAND_PCT = 0.3


def inflation(pack: dict) -> dict:
    infl = pack.get("inflation", {})
    yoy, ann3m = infl.get("core_pce_yoy_pct"), infl.get("core_pce_3m_annualized_pct")
    if not _ok(yoy):
        return na("core PCE unavailable")
    gap = yoy - INFLATION_TARGET_PCT
    trend = ("cooling" if _ok(ann3m) and ann3m < yoy - 0.2
            else "reaccelerating" if _ok(ann3m) and ann3m > yoy + 0.2 else "steady")
    if abs(gap) <= INFLATION_AT_TARGET_BAND_PCT:
        label, tech = "core inflation near the Fed's 2% target", "supportive"
    elif gap > 0:
        label, tech = "core inflation above target", "headwind" if trend != "cooling" else "neutral"
    else:
        label, tech = "core inflation below target", "supportive"
    # Gold's inflation link runs mostly through real yields/policy, not the
    # level itself - a hot-but-cooling print is not the same trade as a
    # hot-and-reaccelerating one, so gold impact keys off trend, not level.
    gold = "supportive" if trend == "reaccelerating" else "neutral"
    return {"applicable": True, "label": f"{label} ({trend})", "tech_impact": tech, "gold_impact": gold,
            "detail": f"core PCE {yoy:.1f}% YoY" + (f", {ann3m:.1f}% 3mo annualized" if _ok(ann3m) else "")}


# +/-15bps/3mo: roughly what a "meaningful" quarter-over-quarter real-yield
# move looks like historically (a 25bp Fed move fully passed through).
REAL_YIELD_CHANGE_THRESHOLD_BPS = 15.0


def real_yields(pack: dict) -> dict:
    """Real yields are the single most textbook-reliable gold driver (higher
    real yield = higher opportunity cost of holding a zero-yield asset) and
    the direct input to every equity discount rate — one number, two seats."""
    ry = pack.get("real_yields", {})
    level = (ry.get("real_10y") or {}).get("value")
    chg_bps = ry.get("real_10y_3m_change_bps")
    if not _ok(level):
        return na("10y real yield (DFII10) unavailable")
    if _ok(chg_bps) and chg_bps > REAL_YIELD_CHANGE_THRESHOLD_BPS:
        label, tech, gold = "real yields rising", "headwind", "headwind"
    elif _ok(chg_bps) and chg_bps < -REAL_YIELD_CHANGE_THRESHOLD_BPS:
        label, tech, gold = "real yields falling", "supportive", "supportive"
    else:
        label, tech, gold = "real yields roughly stable", "neutral", "neutral"
    detail = f"10y real yield {level:.2f}%"
    if _ok(chg_bps):
        detail += f", {chg_bps:+.0f}bps over 3mo"
    return {"applicable": True, "label": label, "tech_impact": tech, "gold_impact": gold, "detail": detail}


# Sahm rule >= 0.5pp has a clean track record calling US recession starts
# (Sahm 2019); yield-curve inversion (10y-2y < 0) is the classic recession
# lead indicator, though with a long and variable lag.
SAHM_RECESSION_THRESHOLD = 0.5


def growth_recession(pack: dict) -> dict:
    gr = pack.get("growth_recession", {})
    sahm = (gr.get("sahm_rule") or {}).get("value")
    curve = (gr.get("yield_curve_10y2y") or {}).get("value")
    unemp = (gr.get("unemployment_rate") or {}).get("value")
    if not _ok(sahm, curve):
        return na("Sahm rule or yield curve unavailable")
    flags = []
    if sahm >= SAHM_RECESSION_THRESHOLD:
        flags.append(f"Sahm rule triggered ({sahm:.2f} >= {SAHM_RECESSION_THRESHOLD})")
    if curve < 0:
        flags.append(f"yield curve inverted ({curve:+.2f}pp)")
    if flags:
        label, tech, gold = "recession-risk signal(s) active: " + "; ".join(flags), "headwind", "supportive"
    else:
        label, tech, gold = "no active recession-risk trigger", "supportive", "neutral"
    detail = f"Sahm {sahm:.2f}, 10y-2y {curve:+.2f}pp"
    if _ok(unemp):
        detail += f", unemployment {unemp:.1f}%"
    return {"applicable": True, "label": label, "tech_impact": tech, "gold_impact": gold, "detail": detail}


# HY spread and VIX bands are standard market-commentary tiers, not a
# proprietary threshold - "calm/elevated/stressed" language matches how the
# Fed's own financial-stability reports describe these same series.
HY_SPREAD_CALM_PCT, HY_SPREAD_STRESSED_PCT = 3.5, 6.0
VIX_CALM, VIX_STRESSED = 15.0, 25.0


def financial_conditions(pack: dict) -> dict:
    fc = pack.get("financial_conditions", {})
    hy = (fc.get("high_yield_spread") or {}).get("value")
    vix = (fc.get("vix") or {}).get("value")
    if not _ok(hy, vix):
        return na("HY spread or VIX unavailable")
    stressed = hy > HY_SPREAD_STRESSED_PCT or vix > VIX_STRESSED
    calm = hy < HY_SPREAD_CALM_PCT and vix < VIX_CALM
    if stressed:
        label, tech, gold = "financial conditions stressed", "headwind", "supportive"
    elif calm:
        label, tech, gold = "financial conditions calm", "supportive", "neutral"
    else:
        label, tech, gold = "financial conditions normal", "neutral", "neutral"
    return {"applicable": True, "label": label, "tech_impact": tech, "gold_impact": gold,
            "detail": f"HY spread {hy:.1f}%, VIX {vix:.1f}"}


# +/-3%/3mo: a broad-dollar-index move of this size is a genuine trend, not
# day-to-day FX noise (DTWEXBGS's own historical volatility is low).
DOLLAR_CHANGE_THRESHOLD_PCT = 3.0


def dollar_fx(pack: dict) -> dict:
    """Dollar strength is a real but secondary tech driver (overseas
    revenue translation) and a more direct gold driver (gold is dollar-
    priced globally) - impacts are asymmetric on purpose."""
    fx = pack.get("dollar_fx", {})
    chg = fx.get("broad_dollar_index_3m_change_pct")
    level = (fx.get("broad_dollar_index") or {}).get("value")
    if not _ok(chg, level):
        return na("broad dollar index unavailable")
    if chg > DOLLAR_CHANGE_THRESHOLD_PCT:
        label, tech, gold = "dollar strengthening", "neutral", "headwind"
    elif chg < -DOLLAR_CHANGE_THRESHOLD_PCT:
        label, tech, gold = "dollar weakening", "neutral", "supportive"
    else:
        label, tech, gold = "dollar roughly flat", "neutral", "neutral"
    return {"applicable": True, "label": label, "tech_impact": tech, "gold_impact": gold,
            "detail": f"broad dollar index {level:.1f}, {chg:+.1f}% over 3mo"}


def event_calendar(pack: dict) -> dict:
    ec = pack.get("event_calendar", {})
    if ec.get("next_fomc_date") is None:
        return na("next FOMC date unavailable")
    binary = ec.get("binary_event_within_3d")
    label = (f"FOMC in {ec['next_fomc_days']} day(s) — avoid new sizing before it"
            if binary else f"next FOMC in {ec['next_fomc_days']} day(s)")
    return {"applicable": True, "label": label, "tech_impact": "neutral", "gold_impact": "neutral",
            "detail": f"next FOMC {ec['next_fomc_date']}. {ec.get('next_nfp_note', '')}"}


ANGLES = {
    "policy_path": policy_path, "liquidity": liquidity, "inflation": inflation,
    "real_yields": real_yields, "growth_recession": growth_recession,
    "financial_conditions": financial_conditions, "dollar_fx": dollar_fx,
    "event_calendar": event_calendar,
}


def scorecard(pack: dict, angles: dict = ANGLES) -> dict:
    """Every angle's rule result plus coverage — the meta-signal that says
    whether the regime/tilt below rests on a full pack or a thin one,
    exactly like valuation.data_coverage_score does for a ticker pack.
    `angles` lets macro_backtest.py score only the historically
    replayable subset, with coverage measured against that subset."""
    results = {name: fn(pack) for name, fn in angles.items()}
    applicable = [r for r in results.values() if r["applicable"]]
    coverage = len(applicable) / len(angles)
    return {"angles": results, "coverage": coverage,
            "coverage_detail": f"{len(applicable)}/{len(angles)} angles resolved"}


# Below this coverage, "add/wait" tilts are printed as "no view" rather
# than trusted - same floor guardrails.py uses for a ticker pack's
# data_coverage before it lets conviction run high.
MIN_COVERAGE_FOR_TILT = 0.5


def _tilt_for(card: dict, impact_key: str) -> dict:
    angles = [a for a in card["angles"].values() if a["applicable"]]
    if card["coverage"] < MIN_COVERAGE_FOR_TILT:
        return {"regime": "unmeasured", "tilt": "hold",
                "reason": f"only {card['coverage_detail']} — too thin to tilt"}
    supportive = sum(1 for a in angles if a[impact_key] == "supportive")
    headwind = sum(1 for a in angles if a[impact_key] == "headwind")
    net = supportive - headwind
    if net >= 2:
        regime, tilt = "supportive", "add"
    elif net <= -2:
        regime, tilt = "headwind", "wait"
    else:
        regime, tilt = "mixed", "hold"
    return {"regime": regime, "tilt": tilt,
            "reason": f"{supportive} supportive, {headwind} headwind, "
                      f"{len(angles) - supportive - headwind} neutral of {len(angles)} resolved angles"}


def regime_and_tilt(card: dict) -> dict:
    """Tech-ETF and gold get independent tilts from the SAME angle set —
    net liquidity/real yields/growth often pull them in opposite directions
    (e.g. rising real yields: headwind for tech, headwind for gold too via
    opportunity cost, but a recession signal flips gold supportive while
    tech stays a headwind), so a single combined regime would hide that."""
    return {"tech": _tilt_for(card, "tech_impact"), "gold": _tilt_for(card, "gold_impact"),
            "honest_limit": ("Macro timing is weak at a 12-month horizon. This tilt is for "
                             "avoiding bad entries and sizing (DCA pace), not calling bottoms/tops.")}


# ---------------------------------------------------------------- FOMC redline

def fomc_redline(pack: dict) -> dict:
    """Sentence-level unified diff between the latest and previous FOMC
    statement - the macro LLM committee's redline reader sees ONLY this
    diff, not either full statement, so its job is "explain what changed"
    rather than "summarize 400 words", matching agents_llm's no-math/cite-
    the-pack posture applied to text instead of numbers."""
    fomc = pack.get("fomc", {})
    latest, prev = fomc.get("latest_text"), fomc.get("prev_text")
    if not latest:
        return {"applicable": False, "reason": "latest FOMC statement unavailable", "diff_lines": []}
    if not prev:
        return {"applicable": False, "reason": "no previous FOMC statement to diff against",
                "diff_lines": []}

    def sentences(text: str) -> list[str]:
        # Negative lookbehind on a single capital letter skips "M." / "K."
        # style middle initials (live-checked: a naive ". "-split shredded
        # "Beth M. Hammack, ... Lorie K. Logan" into three fake sentences).
        return [s.strip() for s in re.split(r"(?<!\b[A-Z])\.\s+(?=[A-Z])", text) if s.strip()]

    diff = list(difflib.unified_diff(sentences(prev), sentences(latest),
                                     fromfile=fomc.get("prev_date", "prev"),
                                     tofile=fomc.get("latest_date", "latest"), lineterm="", n=0))
    changed = [ln for ln in diff if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
    return {"applicable": True, "diff_lines": changed,
            "changed": bool(changed), "latest_date": fomc.get("latest_date"),
            "prev_date": fomc.get("prev_date")}

"""Finance research pipeline orchestrator.

Usage:
  .venv/bin/python pipeline.py TICKER [--horizon-months N] [--no-whatsapp]
  .venv/bin/python pipeline.py MACRO [--no-whatsapp]
  .venv/bin/python pipeline.py MACRO-BACKTEST TICKER [--horizon-months N] [--no-whatsapp]

--horizon-months defaults to 12 (agents_llm.DEFAULT_HORIZON_MONTHS).

Runs detached from any agent turn (see finance-agent.sh). On success: writes
reports/<ticker>-<ts>.md and sends a short WhatsApp summary. On ANY failure:
sends one short error line and exits non-zero. No retries — a failed run ends
cleanly so the next message is never blocked (hard lesson from the bug-watcher).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import agents_llm as A
import anchors as AN
import base_rates as BR
import data
import guardrails as G
import macro as MC
import macro_backtest as MB
import changes as CH
import macro_rules as MR
import macro_storage as MS
import portfolio as PF
import valuation as V

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
PACKS = REPORTS / "packs"
# Optional: E.164 number for the WhatsApp summary. Unset = reports are only written to disk.
WHATSAPP_TO = os.environ.get("FINANCE_AGENT_WHATSAPP_TO", "").strip()
# Bump manually when methodology changes (weights, regime rules, prompts) so
# a saved pack/report can be matched against the code version that produced
# it - without this, re-running backtest/track_accuracy against old packs
# would silently mix results from different methodologies.
PIPELINE_VERSION = "2026.10-p3"
MACRO_SNAPSHOT_MAX_AGE_HOURS = 24.0
DISCLAIMER = ("_Automated research for education only — not financial advice. "
              "Data: Yahoo Finance; models are estimates with wide error bars._")


def fnum(v, nd=1):
    if v is None:
        return "n/a"
    if isinstance(v, float) and abs(v) >= 1e9:
        return f"{v / 1e9:,.{nd}f}B"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def get_macro_snapshot() -> dict:
    """Reuses a snapshot < 24h old; otherwise fetches fresh (~seconds of
    keyless FRED/Fed/Yahoo requests, no LLM) and caches it. Shared by
    `analyze <TICKER>`'s ETF-regime layering and the standalone `macro`
    command so a busy day of tickers doesn't repeat ~15 fetches per ticker."""
    cached = MS.latest_snapshot(MACRO_SNAPSHOT_MAX_AGE_HOURS)
    if cached is not None:
        return cached
    pack = MC.build_macro_pack()
    card = MR.scorecard(pack)
    tilts = MR.regime_and_tilt(card)
    MS.save_snapshot(pack, card, tilts)
    return {"as_of": pack["as_of"], "pack": pack, "card": card, "tilts": tilts, "age_hours": 0.0}


def build_pack(ticker: str, client: A.LLMClient, budget: A.Budget,
               horizon_months: int = A.DEFAULT_HORIZON_MONTHS) -> dict:
    pack = data.gather(ticker)
    pack["horizon_months"] = horizon_months
    f = pack["fundamentals"]
    # quoteSummary's growth_estimates/analyst-target endpoints are the one
    # gap data.py's fast_info/SEC/self-computed-beta fallbacks can't close
    # (no free unauthenticated source exists) - a live web-search agent call
    # is the last resort, gated so it never overrides a real yfinance value
    # and never runs when the ticker itself failed to resolve.
    needs_consensus = (f.get("growth_est") is None or f.get("analyst_target_mean") is None
                       or f.get("eps_fwd") is None)
    if f.get("price") and needs_consensus:
        consensus = A.fetch_market_consensus(client, ticker, f.get("name"), budget)
        cited = ", ".join(consensus.get("consensus_sources", []))
        as_of = consensus.get("consensus_as_of", "date unknown")
        if consensus.get("growth_est") is not None and f.get("growth_est") is None:
            f["growth_est"] = consensus["growth_est"]
            f["growth_est_source"] = f"web search (LLM-summarized, {as_of}, unverified) — {cited}"
        if consensus.get("eps_fwd") is not None and f.get("eps_fwd") is None:
            # unlocks AEG (the only non-fade point model) when quoteSummary's
            # forwardEps is blocked - same unverified-web-search posture as
            # growth_est, never overriding a real yfinance value.
            f["eps_fwd"] = consensus["eps_fwd"]
            f["eps_fwd_source"] = f"web search (LLM-summarized, {as_of}, unverified) — {cited}"
        got_target = False
        for field in ("analyst_target_mean", "analyst_target_median", "analyst_target_count"):
            if consensus.get(field) is not None and f.get(field) is None:
                f[field] = consensus[field]
                got_target = True
        if got_target:
            f["analyst_target_source"] = f"web search (LLM-summarized, {as_of}, unverified) — {cited}"
        if consensus.get("recent_analyst_calls"):
            f["recent_analyst_calls"] = consensus["recent_analyst_calls"]
        if consensus.get("eps_revision_90d"):
            f["eps_revision_90d"] = consensus["eps_revision_90d"]
            f["eps_revision_90d_detail"] = consensus.get("eps_revision_90d_detail")
            f["eps_revision_90d_source"] = f"web search (LLM-summarized, unverified) — {cited}"
    # Free/structured (Yahoo's public earnings-calendar HTML page, not the
    # blocked quoteSummary crumb endpoint) - runs every time, unlike the
    # web-search consensus block above which is gated on missing Yahoo data.
    pack["earnings_calendar"] = data.fetch_earnings_calendar(ticker)
    # Entirely optional (see portfolio.py's module docstring) - no
    # portfolio.yaml means pack["portfolio"] stays None, not an error.
    user_portfolio = PF.load_portfolio()
    pack["portfolio"] = (PF.snapshot(user_portfolio, current_ticker=ticker, current_price=f.get("price"))
                         if user_portfolio else None)
    macro = pack["macro"]
    pack["valuation"] = V.run_valuation(
        f, rf=macro["rf"], erp=macro["erp"],
        rf_source=macro["rf_source"], erp_source=macro["erp_source"])
    # Layer the macro regime into every ticker analysis (not just ETFs) -
    # gold and gold-tracking tickers get the gold tilt, every other ticker
    # gets tech as the general equity-risk-appetite proxy. The "macro"
    # analyst seat below treats this as primary evidence for ETF/fund
    # regimes and secondary context for single stocks (see its persona).
    snap = get_macro_snapshot()
    asset_class = data.macro_asset_class(ticker)
    pack["macro_regime"] = {"asset_class": asset_class, "tilt": snap["tilts"][asset_class],
                            "scorecard": snap["card"], "snapshot_as_of": snap["as_of"]}
    pack["anchors"] = _anchor_percentiles(ticker, f, pack["valuation"], macro["rf"], macro["erp"])
    # No regime gate (unlike anchors, which needs fundamentals) - a price-
    # only base rate applies just as well to ETFs/commodities as equities.
    pack["base_rates"] = BR.signal_base_rates(ticker, horizon_months=horizon_months)
    return pack


def _anchor_percentiles(ticker: str, f: dict, valuation: dict, rf: float, erp: float) -> dict:
    """Model-free own-history percentiles (see anchors.py's module docstring
    for the rationale): a second, no-LLM-math anchor next to the fade-based
    valuation ensemble above, which DESIGN.md's own review flags as prone to
    overstating "overvalued" on quality compounders. ETFs/funds get a price-
    percentile proxy; equities get P/S, EV/FCF, and an implied-growth
    percentile anchored to reverse_dcf's already-solved growth rate (not a
    second solve)."""
    if valuation["regime"]["regime"] == "etf":
        return {"price_percentile": AN.etf_price_percentile(ticker)}
    rd = valuation["models"].get("reverse_dcf", {})
    current_growth = rd["details"].get("implied_growth") if rd.get("applicable") else None
    return {"ps_percentile": AN.ps_percentile(ticker, f),
            "ev_fcf_percentile": AN.ev_fcf_percentile(ticker, f),
            "growth_percentile": AN.growth_percentile(ticker, f, rf, erp, current_growth)}


def _redact_valuation_for_domain_seats(v: dict) -> dict:
    """fundamental/buffett/burry get quality scores, ROCE decomposition, and
    the reverse-DCF implied-vs-Street growth comparison, but NOT the ensemble
    dollar fair value, its sensitivity/Monte Carlo band, or the buyback-vs-
    fair-value flag. Those seats voted bearish (82/74 conviction on AAPL)
    by treating "$315 vs model $48" as a business-quality fact rather than
    a conservative-fade modeling choice the valuation seat is explicitly
    warned about (VAL_NUANCE) - warning one seat and not the other three
    that read the same number is not a fix. Valuation and the trader/risk
    seats still see the full figure."""
    redacted = {k: x for k, x in v.items()
               if k not in ("fair_value", "upside_pct", "sensitivity", "confidence_range")}
    quality = dict(redacted.get("quality", {}))
    bb = quality.get("buyback_value_check")
    if bb and bb.get("applicable"):
        details = {k: x for k, x in bb["details"].items()
                  if k not in ("fair_value", "premium_to_fair_value_pct", "flag_buybacks_above_fair_value")}
        quality["buyback_value_check"] = {**bb, "details": details}
    redacted["quality"] = quality
    return redacted


def pack_for_llm(pack: dict, redact_fair_value: bool = False) -> str:
    """Compact, numbers-first view for prompts (trim statements bulk).
    redact_fair_value=True builds the view for fundamental/buffett/burry
    (see _redact_valuation_for_domain_seats); the rest of the committee
    reads the full pack."""
    f = pack["fundamentals"]
    slim = {
        "ticker": pack["ticker"], "as_of": pack["as_of"],
        "holding_period_months": pack.get("horizon_months", A.DEFAULT_HORIZON_MONTHS),
        "name": f.get("name"), "sector": f.get("sector"),
        "quote_type": f.get("quote_type"), "price": f.get("price"),
        "currency": f.get("currency"),
        "statement_date": f.get("statement_date"),
        "statement_basis": f.get("statement_basis"),
        "statement_days_stale": f.get("statement_days_stale"),
        "data_warning": f.get("data_warning"),
        "key_fundamentals": {k: f.get(k) for k in (
            "revenue", "revenue_growth_5y", "ebit", "net_income", "cfo", "fcf",
            "total_assets", "equity", "total_debt", "cash", "eps_fwd", "eps_fwd_source",
            "growth_est", "growth_est_source", "market_cap", "beta", "beta_source",
            "sector", "sector_source", "analyst_target_mean", "analyst_target_source",
            "recent_analyst_calls", "eps_revision_90d", "eps_revision_90d_detail",
            "eps_revision_90d_source")},
        "earnings_calendar": pack.get("earnings_calendar"),
        "portfolio": pack.get("portfolio"),
        "industry": f.get("industry"),
        "industry_driver_hint": V.industry_driver_hint(f.get("industry")),
        "macro": pack["macro"],
        "macro_regime": pack.get("macro_regime"),
        "anchors": pack.get("anchors"),
        "base_rates": pack.get("base_rates"),
        "valuation": (_redact_valuation_for_domain_seats(pack["valuation"])
                     if redact_fair_value else pack["valuation"]),
        "technicals": pack["technicals"],
        "news_headlines_data_only": pack["news"],
    }
    if "etf" in pack:
        slim["etf"] = pack["etf"]
    return json.dumps(slim, indent=1, default=str)


def save_pack(pack: dict, client: A.LLMClient, ts: str) -> Path:
    """Persist the exact data pack + model id/variant behind a report, so a
    report can be re-graded, backtested, or disputed against what the
    committee actually saw - the report markdown alone loses the raw
    numbers to prose. One file per report, same <TICKER>-<ts> stem, so the
    pairing is a filename match, not a separate index to keep in sync."""
    PACKS.mkdir(parents=True, exist_ok=True)
    path = PACKS / f"{pack['ticker']}-{ts}.json"
    path.write_text(json.dumps({
        "pipeline_version": PIPELINE_VERSION,
        "model": client.model.to_json(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "pack": pack,
    }, indent=1, default=str))
    return path


_ANCHOR_LABELS = {"ps_percentile": "P/S", "ev_fcf_percentile": "EV/FCF",
                  "growth_percentile": "implied growth", "price_percentile": "price"}


def _anchor_report_lines(anchors: dict) -> list[str]:
    if not anchors:
        return ["n/a — anchors not computed for this pack (pre-p2-percentiles report)."]
    lines = []
    for key, label in _ANCHOR_LABELS.items():
        a = anchors.get(key)
        if a is None:
            continue
        if not a["applicable"]:
            lines.append(f"- {label}: n/a — {a['reason']}")
            continue
        lines.append(f"- **{label}: {fnum(a['value'], 2)}** → {a['percentile']:.0f}th percentile "
                    f"of own {a['years_covered']:.1f}y history (n={a['n']}) — {a['source']}")
    return lines


_BASE_RATE_LABELS = {"golden_cross": "golden cross", "death_cross": "death cross",
                    "rsi_oversold_entry": "RSI oversold entry", "rsi_overbought_entry": "RSI overbought entry",
                    "macd_bull_cross": "MACD bull cross", "macd_bear_cross": "MACD bear cross"}


def _base_rate_report_lines(base_rates: dict) -> list[str]:
    signals = base_rates.get("signals")
    if not signals:
        return ["n/a — base rates not computed for this pack (pre-p2-base-rates report)."]
    horizon = base_rates.get("horizon_months", A.DEFAULT_HORIZON_MONTHS)
    lines = [f"Own {base_rates['years_covered']}y history, {horizon}-month forward window:"]
    for key, label in _BASE_RATE_LABELS.items():
        s = signals.get(key)
        if s is None:
            continue
        if not s["applicable"]:
            lines.append(f"- {label}: never fired with a full {horizon}mo forward window "
                        f"({s['n_fired']} raw occurrence(s))")
            continue
        lo, hi = s["hit_rate_95ci"]
        trust = "" if s["trusted"] else " ⚠️ small sample, do not treat as proof"
        lines.append(f"- **{label}**: {s['hit_rate'] * 100:.0f}% hit rate "
                    f"(95% CI {lo * 100:.0f}–{hi * 100:.0f}%, n={s['n']}, "
                    f"avg fwd return {s['avg_forward_return_pct']:+.1f}%){trust}")
    return lines


def _portfolio_report_lines(pf: dict, ticker: str) -> list[str]:
    if not pf["applicable"]:
        return [f"n/a — {pf['reason']}"]
    cur = pf.get("currency") or ""
    lines = [f"Total value: **{fnum(pf['total_value'], 0)}** {cur} · "
            f"cash {pf['cash_pct']}% · concentration HHI **{pf['concentration_hhi']}** "
            "(0 = fully diversified, 10000 = one position)",
            f"Existing {ticker} weight before this call: **{pf['current_ticker_existing_pct']}%**"]
    for t, w in sorted(pf["holdings_pct"].items(), key=lambda kv: -kv[1]):
        target = pf["targets_pct"].get(t)
        drift = f" (target {target}%, drift {w - target:+.1f}pp)" if target is not None else ""
        lines.append(f"- {t}: {w}%{drift}")
    if pf["targets_pct"].get("cash") is not None:
        drift = pf["cash_pct"] - pf["targets_pct"]["cash"]
        lines.append(f"- cash: {pf['cash_pct']}% (target {pf['targets_pct']['cash']}%, drift {drift:+.1f}pp)")
    if pf["unpriced_holdings"]:
        lines.append(f"⚠️ Could not price (excluded from total): {', '.join(pf['unpriced_holdings'])}")
    return lines


def _macro_scorecard_lines(mr: dict) -> list[str]:
    """Per-angle table from the saved scorecard (not model prose), so every
    angle's label/impact is visible, including unresolved ones."""
    card = (mr or {}).get("scorecard") or {}
    if not card.get("angles"):
        return []
    key = "gold_impact" if mr.get("asset_class") == "gold" else "tech_impact"
    t = mr.get("tilt") or {}
    lines = ["", f"## Macro scorecard ({mr.get('asset_class')}) — tilt {t.get('tilt')}, "
             f"{card.get('coverage_detail')}", "",
             "| Angle | Impact | Reading | Detail |", "|---|---|---|---|"]
    for name, a in card["angles"].items():
        if a.get("applicable"):
            lines.append(f"| {name} | {a.get(key)} | {a.get('label')} | {a.get('detail')} |")
        else:
            lines.append(f"| {name} | unresolved | — | {a.get('reason')} |")
    return lines


def _invalidation_level_lines(d: dict) -> list[str]:
    """Machine-readable line track_accuracy.INVALIDATION_RE parses to check
    whether the stated invalidation actually triggered."""
    lv = d.get("invalidation_level")
    return [f"Invalidation level: close {lv['direction']} {lv['price']:.2f}"] if lv else []


def _committee_check_lines(d: dict) -> list[str]:
    out = []
    ja = d.get("judge_agreement")
    if ja:
        out.append("Two judge runs: " + (" / ".join(s.upper() for s in ja["signals"]))
                   + (" (agree)" if ja["agree"] else " (DISAGREE)"))
    ab = d.get("ablation")
    if ab:
        out.append(f"Ablation (no dollar fair value): {ab['signal'].upper()} "
                   f"conviction {ab['conviction']}" + (" - FLIPPED" if ab["flipped"] else " - same call"))
    return out


def write_report(pack, analysts, decision, ts: str, pack_path: Path | None,
                 model_id: str, changes: dict | None = None) -> Path:
    REPORTS.mkdir(exist_ok=True)
    tk = pack["ticker"]
    path = REPORTS / f"{tk}-{ts}.md"
    v = pack["valuation"]
    d = decision["decision"]
    lines = [
        f"# {tk} — {pack['fundamentals'].get('name')} — {pack['as_of']}",
        "",
        f"**FINAL: {d['action']}** · conviction {d['conviction']}/100 · "
        f"size {d['size_pct']}% · horizon {d['horizon_months']}mo",
        f"Invalidation: {d['invalidation']}",
        *_invalidation_level_lines(d),
        *_committee_check_lines(d),
        f"Rationale: {d['rationale']}",
        f"Pipeline {PIPELINE_VERSION} · model {model_id}"
        + (f" · pack: {pack_path.relative_to(BASE)}" if pack_path else ""),
    ]
    if d.get("guardrail_notes"):
        lines.append("Guardrail overrides:")
        lines += [f"  - {n}" for n in d["guardrail_notes"]]
    if changes:
        lines += ["", f"## What changed since {changes['since']}"] + [f"- {c}" for c in changes["lines"]]
    lines += [
        "",
        f"## Regime: {v['regime']['regime']} — {v['regime']['reason']}",
        f"Statements used: {pack['fundamentals'].get('statement_date') or 'n/a'}"
        + (f" ({pack['fundamentals']['statement_basis']}, "
           f"{pack['fundamentals']['statement_days_stale']} days stale)"
           if pack['fundamentals'].get('statement_days_stale') is not None else "")
        + (f" · ⚠️ {pack['fundamentals'].get('data_warning')}"
           if pack['fundamentals'].get('data_warning') else ""),
        f"Rates: rf {fnum(v['rates_used']['rf'] * 100, 2)}% ({v['rates_used']['rf_source']}) · "
        f"ERP {fnum(v['rates_used']['erp'] * 100, 2)}% ({v['rates_used']['erp_source']})",
        # Printed unconditionally (unlike the ensemble line below, which only
        # exists when a valuation model applies) so track_accuracy can score
        # EVERY report against price action, including ETFs/ETCs that have
        # no applicable model - the gap that made EQQQ.L/IGLN.L unscoreable.
        f"Price at report: **{fnum(pack['fundamentals'].get('price'), 2)}** "
        f"{pack['fundamentals'].get('currency') or ''}".rstrip(),
    ]
    ec = pack.get("earnings_calendar") or {}
    if ec.get("next_earnings_date"):
        flag = " ⚠️ within 2 weeks" if ec.get("flag_earnings_within_2w") else ""
        lines.append(f"Next earnings: **{ec['next_earnings_date']}** "
                    f"({ec['next_earnings_days_away']}d away, consensus EPS "
                    f"{fnum(ec.get('next_earnings_eps_estimate'), 2)}){flag}")
    if ec.get("flag_recent_earnings_misses"):
        lines.append("⚠️ Missed consensus EPS in each of the last 2 reported quarters")
    if ec.get("recent_earnings_surprises"):
        surprises = ", ".join(f"{s['date']}: {fnum(s['surprise_pct'], 1)}%"
                              for s in ec["recent_earnings_surprises"])
        lines.append(f"Recent EPS surprises: {surprises}")
    if (pack["fundamentals"].get("growth_est_source") or "").startswith("web search"):
        lines.append(f"⚠️ growth_est via {pack['fundamentals']['growth_est_source']}")
    if (pack["fundamentals"].get("eps_fwd_source") or "").startswith("web search"):
        lines.append(f"⚠️ eps_fwd via {pack['fundamentals']['eps_fwd_source']}")
    if pack["fundamentals"].get("eps_revision_90d"):
        detail = pack["fundamentals"].get("eps_revision_90d_detail")
        lines.append(f"⚠️ EPS estimate revisions (90d, unverified): "
                    f"**{pack['fundamentals']['eps_revision_90d'].upper()}**"
                    + (f" — {detail}" if detail else ""))
    lines += _macro_scorecard_lines(pack.get("macro_regime") or {})
    lines += [
        "",
        "## Valuation models",
    ]
    for n, m in v["models"].items():
        if m["applicable"]:
            det = {k: (round(x, 4) if isinstance(x, float) else x)
                   for k, x in m["details"].items()}
            val = f" → **{fnum(m['value'], 2)}/sh**" if m["value"] is not None else ""
            lines.append(f"- **{n}**{val} — {det}")
        else:
            lines.append(f"- {n}: excluded — {m['reason']}")
    rd = v["models"].get("reverse_dcf", {})
    if rd.get("applicable"):
        ig = rd["details"].get("implied_growth")
        hg, street = rd["details"].get("hist_growth_5y"), rd["details"].get("street_growth_est")
        wacc_used, tg_used = rd["details"].get("wacc"), rd["details"].get("terminal_g")
        # Street comparison is the unbiased pair the fade-based dollar figure
        # is not: both implied and Street growth are read off the same
        # market/consensus inputs, with no competitive-fade assumption.
        street_bit = (f" vs Street consensus {fnum(street * 100, 1)}%/yr"
                     if street is not None else "")
        lines += ["",
                  f"**Price implies {ig * 100:.1f}%/yr growth** ({rd['details']['base_kind']} "
                  f"base, WACC {fnum((wacc_used or 0) * 100, 1)}%, terminal g "
                  f"{fnum((tg_used or 0) * 100, 1)}%) vs {fnum((hg or 0) * 100, 1)}% "
                  f"historical 5y{street_bit} — a falsifiable, assumption-light read; "
                  "see the blended dollar figure below for context, not as the headline claim."]
    if v["fair_value"] is not None:
        s = v["sensitivity"] or {}
        cr = v.get("confidence_range") or {}
        band = (f" · Monte Carlo band [{fnum(cr['details']['p10'], 2)} — {fnum(cr['details']['p90'], 2)}] "
               f"(p10-p90, n={cr['details']['n']})" if cr.get("applicable") else "")
        lines += ["",
                  f"Weighted ensemble (context, not the headline): **{fnum(v['fair_value'], 2)}** vs price "
                  f"{fnum(pack['fundamentals'].get('price'), 2)} "
                  f"(upside {fnum(v['upside_pct'], 1)}%) · ±1% rate band "
                  f"[{fnum(s.get('rate_plus_1pct'), 2)} — {fnum(s.get('rate_minus_1pct'), 2)}]{band}"]
    at = v.get("analyst_target", {})
    if at.get("applicable"):
        src = pack["fundamentals"].get("analyst_target_source")
        lines.append(f"Street consensus target (context only, not blended in): "
                    f"**{fnum(at['value'], 2)}** ({fnum(at['details'].get('target_vs_price_pct'), 1)}% "
                    f"vs price, {fnum(at['details'].get('analyst_count'), 0)} analysts)"
                    + (f" — ⚠️ {src}" if src and src.startswith("web search") else ""))
    calls = pack["fundamentals"].get("recent_analyst_calls")
    if calls:
        lines.append("")
        lines.append("Recent named analyst calls (from the sources above, not a full panel):")
        for c in calls:
            who = f"{c['analyst']} ({c['firm']})" if c.get("analyst") else c["firm"]
            bits = [b for b in (c.get("action"), c.get("rating")) if b]
            pt = f" → {fnum(c['price_target'], 2)}" if c.get("price_target") is not None else ""
            when = f" [{c['date']}]" if c.get("date") else ""
            lines.append(f"- {who}: {' '.join(bits) or 'n/a'}{pt}{when}")
    dc = v.get("data_coverage", {})
    if dc.get("applicable"):
        lines.append(f"Data coverage: **{dc['value'] * 100:.0f}%** "
                    f"({dc['details']['present']}/{dc['details']['of']} fundamental fields populated)")
    lines += ["", "## Own-history anchors (model-free)", ""]
    lines += _anchor_report_lines(pack.get("anchors") or {})
    lines += ["", "## Signal base rates (model-free)", ""]
    lines += _base_rate_report_lines(pack.get("base_rates") or {})
    if pack.get("portfolio") is not None:
        lines += ["", "## Portfolio context", ""]
        lines += _portfolio_report_lines(pack["portfolio"], tk)
    lines += ["", "## Quality overlay"]
    for qn, q in {**v["quality"], "roce_decomposition": v["roce_decomposition"]}.items():
        if q["applicable"]:
            flag = "" if not any(k.startswith("flag") and x for k, x in q["details"].items()) else " ⚠️"
            lines.append(f"- {qn}: **{fnum(q['value'], 2)}**{flag} {q['details']}")
        else:
            lines.append(f"- {qn}: n/a — {q['reason']}")
    if v.get("quick_ratios"):
        lines.append(f"- quick_ratios: {v['quick_ratios']}")
    lines += ["", "## Analyst votes"]
    for name, a in analysts.items():
        cit = a.get("citations", {})
        cit_bit = ""
        if cit.get("total"):
            cit_bit = f" [cited {cit['verified']}/{cit['total']}"
            if cit.get("unverified"):
                cit_bit += f", unverified: {cit['unverified']}"
            cit_bit += "]"
        lines.append(f"- **{name}**: {a['signal']} ({a['confidence']}) — {a['reasoning']}{cit_bit}")
    lines += ["", "## Trader: bull vs bear and plan", "", decision["trader_text"], "",
              "## Risk seats", ""]
    lines += [t + "\n" for t in decision["risk_transcript"]]
    lines += [f"**Risk judge:** {decision['judge_text']}", "", "---", DISCLAIMER]
    path.write_text("\n".join(lines))
    return path


def short_summary(pack, analysts, decision, report_path) -> str:
    v, d = pack["valuation"], decision["decision"]
    f = pack["fundamentals"]
    emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}[d["signal"]]
    votes = ", ".join(f"{k}:{a['signal'][:4]}" for k, a in analysts.items())
    out = [f"📊 {pack['ticker']} ({f.get('name')}) — {emoji} {d['action']} "
           f"(conviction {d['conviction']}, size {d['size_pct']}%, {d['horizon_months']}mo)",
           f"Price {fnum(f.get('price'), 2)} · regime: {v['regime']['regime']}"]
    # Tier 3: lead with reverse-DCF implied growth (falsifiable, assumption-
    # light) instead of the blended dollar figure (a conservative-knob mash).
    rd = v["models"].get("reverse_dcf", {})
    if rd.get("applicable"):
        ig, hg = rd["details"].get("implied_growth"), rd["details"].get("hist_growth_5y")
        street = rd["details"].get("street_growth_est")
        street_bit = f" (Street {fnum(street * 100, 0)}%)" if street is not None else ""
        out.append(f"Price implies {ig * 100:.0f}%/yr growth vs {fnum((hg or 0) * 100, 1)}% hist.{street_bit}")
    if v["fair_value"] is not None:
        out.append(f"Ensemble (context): {fnum(v['fair_value'], 2)} ({fnum(v['upside_pct'], 1)}% vs price)")
    q = v["quality"]
    qbits = []
    if q["piotroski_f"]["applicable"]:
        qbits.append(f"F={q['piotroski_f']['value']}")
    if q["mohanram_g"]["applicable"]:
        qbits.append(f"G={q['mohanram_g']['value']}")
    if q["altman_z"]["applicable"]:
        qbits.append(f"Z={q['altman_z']['value']:.1f}({q['altman_z']['details']['zone']})")
    if qbits:
        out.append("Quality: " + " ".join(qbits))
    out.append(f"Votes: {votes}")
    out.append(f"Why: {d['rationale'][:160]}")
    out.append(f"Invalidation: {d['invalidation'][:120]}")
    if d.get("guardrail_notes"):
        out.append(f"⚠️ Guardrail overrides applied ({len(d['guardrail_notes'])}) — see report")
    out.append(f"📄 Full report: {report_path}")
    out.append("⚠️ Research only, not financial advice.")
    return "\n".join(out)


def write_macro_report(snap: dict, committee: dict, ts: str) -> Path:
    prev = MS.snapshot_before(snap["pack"]["as_of"])
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"MACRO-{ts}.md"
    card, tilts, pack = snap["card"], snap["tilts"], snap["pack"]
    lines = [
        f"# Macro regime — {pack['as_of']}",
        "",
        f"**Tech ETF tilt: {tilts['tech']['tilt'].upper()}** ({tilts['tech']['regime']}) "
        f"— {tilts['tech']['reason']}",
        f"**Gold tilt: {tilts['gold']['tilt'].upper()}** ({tilts['gold']['regime']}) "
        f"— {tilts['gold']['reason']}",
        tilts["honest_limit"],
        f"Pipeline {PIPELINE_VERSION} · coverage {card['coverage_detail']}",
        "", "## Scorecard",
    ]
    for name, a in card["angles"].items():
        if a["applicable"]:
            lines.append(f"- **{name}**: {a['label']} — tech:{a['tech_impact']} "
                        f"gold:{a['gold_impact']} — {a['detail']}")
        else:
            lines.append(f"- {name}: n/a — {a['reason']}")
    if prev is not None:
        flips = CH.diff_macro(prev["card"], card, prev["tilts"], tilts)
        lines += ["", f"## What changed since the previous snapshot ({prev['as_of']})"]
        lines += [f"- {c}" for c in flips] or ["- no angle label or tilt changed"]
    lines += ["", "## FOMC redline", "", committee["redline_text"]]
    if pack.get("fomc", {}).get("latest_url"):
        lines.append(f"Source: {pack['fomc']['latest_url']}")
    lines += ["", "## Fiscal & trade (web search, unverified)", ""]
    ft = committee["fiscal_trade"]
    if ft.get("fiscal_summary"):
        lines.append(f"Fiscal: {ft['fiscal_summary']}")
    if ft.get("trade_summary"):
        lines.append(f"Trade: {ft['trade_summary']}")
    if not ft.get("fiscal_summary") and not ft.get("trade_summary"):
        lines.append("(no verifiable web-search result this run)")
    if ft.get("sources"):
        lines.append("Sources: " + ", ".join(ft["sources"]))
    lines += ["", "## Tech judge", "", committee["tech_judge"]["text"],
             "", "## Gold judge", "", committee["gold_judge"]["text"],
             "", "## Devil's advocate", "", committee["devils_advocate"],
             "", "---", DISCLAIMER]
    path.write_text("\n".join(lines))
    return path


def macro_short_summary(snap: dict, card: dict, report_path: Path) -> str:
    tilts = snap["tilts"]
    emoji = {"add": "🟢", "hold": "🟡", "wait": "🔴"}
    out = [
        f"🌐 Macro regime — {snap['as_of']}",
        f"{emoji.get(tilts['tech']['tilt'], '🟡')} Tech ETF: {tilts['tech']['tilt'].upper()} "
        f"— {tilts['tech']['reason']}",
        f"{emoji.get(tilts['gold']['tilt'], '🟡')} Gold: {tilts['gold']['tilt'].upper()} "
        f"— {tilts['gold']['reason']}",
    ]
    ec = card["angles"].get("event_calendar", {})
    if ec.get("applicable"):
        out.append(f"📅 {ec['label']}")
    headwinds = [n for n, a in card["angles"].items()
                if a.get("tech_impact") == "headwind" or a.get("gold_impact") == "headwind"]
    if headwinds:
        out.append("Watch: " + ", ".join(headwinds))
    out.append(tilts["honest_limit"])
    out.append(f"📄 Full report: {report_path}")
    out.append("⚠️ Research only, not financial advice.")
    return "\n".join(out)


def send_whatsapp(text: str):
    if not WHATSAPP_TO:
        return
    subprocess.run(
        ["openclaw", "message", "send", "--channel", "whatsapp",
         "--target", WHATSAPP_TO, "-m", text],
        check=True, capture_output=True, timeout=60)


def _fail(msg: str, notify: bool) -> None:
    """Shared failure path for both the ticker and macro flows: one error
    line to stderr, best-effort WhatsApp, exit 1. No retries — a failed run
    ends cleanly so the next message is never blocked."""
    print(msg, file=sys.stderr)
    if notify:
        try:
            send_whatsapp(msg)
        except Exception:  # noqa: BLE001 - already failing; report on stderr only
            pass
    sys.exit(1)


def _changes_since_last(pack: dict, current_stem: str) -> dict | None:
    prev = CH.previous_ticker_pack(pack["ticker"], current_stem)
    if prev is None:
        return None
    old_pack, stem, old_report = prev
    lines = CH.diff_packs(old_pack, pack)
    if old_report is not None:
        lines.insert(0, f"previous decision: {old_report.signal.upper()} "
                        f"conviction {old_report.conviction}, size {old_report.size_pct}%")
    return {"since": f"last run ({stem.split('-', 1)[1]} UTC)",
            "lines": lines or ["no key field moved"]}


def run_ticker(ticker: str, notify: bool, horizon_months: int = A.DEFAULT_HORIZON_MONTHS) -> None:
    budget = A.Budget()
    try:
        client = A.llm_client()
        pack = build_pack(ticker, client, budget, horizon_months)
        if not pack["fundamentals"].get("price"):
            raise A.PipelineError(f"no market data for {ticker} (bad ticker?)")
        pj = pack_for_llm(pack)
        pj_redacted = pack_for_llm(pack, redact_fair_value=True)
        analysts = A.run_analysts(client, pack, pj, pj_redacted, budget)
        decision = A.run_decision(client, pj, analysts, budget, horizon_months,
                                  pack_json_redacted=pj_redacted)
        decision["decision"] = G.apply_guardrails(pack, analysts, decision["decision"])
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        pack_path = save_pack(pack, client, ts)
        report = write_report(pack, analysts, decision, ts=ts,
                              pack_path=pack_path, model_id=client.model.id,
                              changes=_changes_since_last(pack, f"{ticker}-{ts}"))
        summary = short_summary(pack, analysts, decision, report)
        print(summary)
        if notify:
            send_whatsapp(summary)
    except A.PipelineError as e:
        _fail(f"❌ {ticker} analysis failed: {e}", notify)
    except Exception as e:  # noqa: BLE001
        _fail(f"❌ {ticker} analysis crashed: {e.__class__.__name__}: {str(e)[:140]}", notify)


def run_macro(notify: bool) -> None:
    budget = A.Budget()
    try:
        client = A.llm_client()
        snap = get_macro_snapshot()
        redline = MR.fomc_redline(snap["pack"])
        committee = A.run_macro_committee(client, redline, snap["pack"]["policy_path"],
                                          snap["card"], snap["tilts"], budget)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        report = write_macro_report(snap, committee, ts)
        summary = macro_short_summary(snap, snap["card"], report)
        print(summary)
        if notify:
            send_whatsapp(summary)
    except A.PipelineError as e:
        _fail(f"❌ macro analysis failed: {e}", notify)
    except Exception as e:  # noqa: BLE001
        _fail(f"❌ macro analysis crashed: {e.__class__.__name__}: {str(e)[:140]}", notify)


_BACKTEST_REGIME_ORDER = ("supportive", "mixed", "headwind", "unmeasured")


def _backtest_window_lines(title: str, w: dict) -> list[str]:
    if not w["applicable"]:
        return [f"## {title}", "", f"n/a — {w['reason']}", ""]
    b = w["baseline"]
    lo, hi = b["hit_rate_95ci"]
    lines = [f"## {title}: {w['start']} → {w['end']} (month-ends with a full forward window)", "",
             "| Regime | Months | Separate periods | Hit rate (95% CI) | Avg return | vs all months |",
             "|---|---|---|---|---|---|",
             f"| **all months (baseline)** | {b['n_months']} | — | {b['hit_rate']:.0%} "
             f"({lo:.0%}–{hi:.0%}) | {b['avg_return_pct']:+.1f}% | — |"]
    for label in _BACKTEST_REGIME_ORDER:
        s = w["regimes"].get(label)
        if s is None:
            continue
        lo, hi = s["hit_rate_95ci"]
        flag = "" if s["trusted"] else " ⚠️ too few periods"
        lines.append(f"| {label}{flag} | {s['n_months']} | {s['n_episodes']} | {s['hit_rate']:.0%} "
                     f"({lo:.0%}–{hi:.0%}) | {s['avg_return_pct']:+.1f}% | "
                     f"{s['vs_baseline_return_pp']:+.1f}pp return, {s['vs_baseline_hit_rate_pp']:+.1f}pp hit |")
    return lines + [""]


def write_macro_backtest_report(result: dict, ts: str) -> Path:
    REPORTS.mkdir(exist_ok=True)
    tk, h = result["ticker"], result["horizon_months"]
    path = REPORTS / f"MACRO-BACKTEST-{tk}-{h}m-{ts}.md"
    latest = result["latest_regime"]
    lines = [f"# Macro tilt backtest — {tk} ({result['asset_class']} regime), {h}-month forward return",
             "",
             f"Latest month-end regime ({latest['as_of']}): **{latest['regime']}**",
             f"Pipeline {PIPELINE_VERSION} · price history from {result['price_history_start']}",
             "",
             "Read each regime against the all-months baseline, not against zero. A regime "
             f"with fewer than {result['min_episodes_trusted']} separate periods is too thin "
             "to trust, however many months it covers (a break of up to "
             f"{MB.EPISODE_MERGE_GAP_MONTHS} months does not start a new period).",
             ""]
    lines += _backtest_window_lines("Full window", result["windows"]["full"])
    lines += _backtest_window_lines("Recent window", result["windows"]["recent"])
    first = result["angle_first_applicable"]
    lines += ["## Method and limits", "",
              "- Angles replayed (same rules as the live `macro` command): "
              + ", ".join(f"{a} (from {first.get(a, 'never')})" for a in result["angles_used"]),
              "- Excluded: " + "; ".join(f"{a} — {why}" for a, why in result["angles_excluded"].items()),
              "- Each month-end only sees data public by then (per-series publication delay).",
              "- Uses today's revised FRED values, not the first-published ones.",
              "- If the full and recent windows disagree, treat the result as fragile.",
              "", "---", DISCLAIMER]
    path.write_text("\n".join(lines))
    return path


def macro_backtest_short_summary(result: dict, report_path: Path) -> str:
    out = [f"📈 Macro backtest — {result['ticker']} ({result['asset_class']} regime), "
           f"{result['horizon_months']}mo forward",
           f"Latest regime ({result['latest_regime']['as_of']}): {result['latest_regime']['regime']}"]
    for title, key in (("Full", "full"), ("Recent", "recent")):
        w = result["windows"][key]
        if not w["applicable"]:
            out.append(f"{title}: n/a — {w['reason']}")
            continue
        b = w["baseline"]
        out.append(f"{title} {w['start']}→{w['end']}: all months {b['hit_rate']:.0%} up, "
                   f"avg {b['avg_return_pct']:+.1f}%")
        for label in ("supportive", "headwind"):
            s = w["regimes"].get(label)
            if s:
                flag = "" if s["trusted"] else " ⚠️ thin"
                out.append(f"  {label}: {s['vs_baseline_return_pp']:+.1f}pp vs all months "
                           f"({s['n_episodes']} periods){flag}")
    out.append(f"📄 Full report: {report_path}")
    out.append("⚠️ Research only, not financial advice.")
    return "\n".join(out)


def run_macro_backtest(ticker: str, horizon_months: int, notify: bool) -> None:
    try:
        result = MB.tilt_forward_returns(ticker, horizon_months)
        if result["price_history_start"] is None:
            raise A.PipelineError(f"no price history for {ticker} (bad ticker?)")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        report = write_macro_backtest_report(result, ts)
        summary = macro_backtest_short_summary(result, report)
        print(summary)
        if notify:
            send_whatsapp(summary)
    except A.PipelineError as e:
        _fail(f"❌ {ticker} macro backtest failed: {e}", notify)
    except Exception as e:  # noqa: BLE001
        _fail(f"❌ {ticker} macro backtest crashed: {e.__class__.__name__}: {str(e)[:140]}", notify)


USAGE = ("usage: pipeline.py TICKER [--horizon-months N] [--no-whatsapp]\n"
         "       pipeline.py MACRO [--no-whatsapp]\n"
         "       pipeline.py MACRO-BACKTEST TICKER [--horizon-months N] [--no-whatsapp]")
MAX_HORIZON_MONTHS = 60  # base_rates/backtest history is ~15-20y; longer windows leave too few points


def _usage_exit() -> None:
    print(USAGE)
    sys.exit(2)


def _horizon_arg(argv: list[str]) -> int:
    if "--horizon-months" not in argv:
        return A.DEFAULT_HORIZON_MONTHS
    i = argv.index("--horizon-months")
    value = argv[i + 1] if i + 1 < len(argv) else ""
    if not value.isdigit() or not 1 <= int(value) <= MAX_HORIZON_MONTHS:
        print(f"--horizon-months must be a whole number from 1 to {MAX_HORIZON_MONTHS}")
        _usage_exit()
    return int(value)


def _valid_ticker(arg: str) -> bool:
    # Yahoo symbols use . - = ^ (EQQQ.L, BRK-B, GC=F, ^VIX).
    return arg.translate(str.maketrans("", "", ".-=^")).isalnum()


def main():
    argv = sys.argv[1:]
    notify = "--no-whatsapp" not in argv
    horizon = _horizon_arg(argv)
    positional = [a.upper() for i, a in enumerate(argv)
                  if not a.startswith("--") and (i == 0 or argv[i - 1] != "--horizon-months")]
    cmd = positional[0] if positional else None
    if cmd == "MACRO":
        run_macro(notify)
    elif cmd == "MACRO-BACKTEST":
        if len(positional) < 2 or not _valid_ticker(positional[1]):
            _usage_exit()
        run_macro_backtest(positional[1], horizon, notify)
    elif cmd and _valid_ticker(cmd):
        run_ticker(cmd, notify, horizon)
    else:
        _usage_exit()


if __name__ == "__main__":
    main()

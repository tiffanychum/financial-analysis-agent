"""'What changed since last run': a deterministic diff of the key pack fields
(and the previous decision) against the previous saved pack for the same
ticker, and of macro angle labels/tilts against the previous macro snapshot.
No model calls: both inputs are already persisted (reports/packs/*.json, the
macro_snapshots table)."""

from __future__ import annotations

import json
from pathlib import Path

import track_accuracy as TA

PACKS = Path(__file__).resolve().parent / "reports" / "packs"
REPORTS = PACKS.parent


def _dig(d: dict, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


# (label, path in pack, decimals, scale, suffix)
_FIELDS = (
    ("price", ("fundamentals", "price"), 2, 1, ""),
    ("fair value", ("valuation", "fair_value"), 2, 1, ""),
    ("implied growth", ("valuation", "models", "reverse_dcf", "details", "implied_growth"), 1, 100, "%"),
    ("Street target", ("fundamentals", "analyst_target_mean"), 2, 1, ""),
    ("fwd EPS", ("fundamentals", "eps_fwd"), 2, 1, ""),
    ("P/S percentile", ("anchors", "ps_percentile", "percentile"), 0, 1, ""),
    ("EV/FCF percentile", ("anchors", "ev_fcf_percentile", "percentile"), 0, 1, ""),
    ("growth percentile", ("anchors", "growth_percentile", "percentile"), 0, 1, ""),
    ("RSI14", ("technicals", "rsi14"), 0, 1, ""),
    ("data coverage", ("valuation", "data_coverage", "value"), 0, 100, "%"),
)


def previous_ticker_pack(ticker: str, current_stem: str) -> tuple[dict, str, object] | None:
    """(pack, stem, previous ParsedReport or None) of the newest earlier pack."""
    older = sorted(p for p in PACKS.glob(f"{ticker}-*.json") if p.stem < current_stem)
    if not older:
        return None
    path = older[-1]
    pack = json.loads(path.read_text())["pack"]
    report_md = REPORTS / f"{path.stem}.md"
    return pack, path.stem, (TA.parse_report(report_md) if report_md.exists() else None)


def _fmt(v: float, decimals: int, scale: float, suffix: str) -> str:
    return f"{v * scale:.{decimals}f}{suffix}"


def diff_packs(old: dict, new: dict) -> list[str]:
    lines = []
    for label, path, decimals, scale, suffix in _FIELDS:
        a, b = _dig(old, *path), _dig(new, *path)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if _fmt(a, decimals, scale, suffix) == _fmt(b, decimals, scale, suffix):
            continue
        pct = f" ({(b / a - 1) * 100:+.1f}%)" if a and decimals and suffix == "" else ""
        lines.append(f"{label}: {_fmt(a, decimals, scale, suffix)} → {_fmt(b, decimals, scale, suffix)}{pct}")
    old_t, new_t = _dig(old, "macro_regime", "tilt"), _dig(new, "macro_regime", "tilt")
    if old_t and new_t and (old_t.get("tilt"), old_t.get("regime")) != (new_t.get("tilt"), new_t.get("regime")):
        lines.append(f"macro tilt: {old_t['regime']}/{old_t['tilt']} → {new_t['regime']}/{new_t['tilt']}")
    return lines


def diff_macro(old_card: dict, new_card: dict, old_tilts: dict, new_tilts: dict) -> list[str]:
    lines = []
    for name, new_a in new_card["angles"].items():
        old_a = old_card["angles"].get(name)
        if old_a is None:
            continue
        old_l = old_a["label"] if old_a["applicable"] else "n/a"
        new_l = new_a["label"] if new_a["applicable"] else "n/a"
        if old_l != new_l:
            lines.append(f"{name}: {old_l} → {new_l}")
    for cls in ("tech", "gold"):
        o, n = old_tilts.get(cls, {}), new_tilts.get(cls, {})
        if o.get("tilt") != n.get("tilt"):
            lines.append(f"{cls} tilt: {str(o.get('tilt')).upper()} → {str(n.get('tilt')).upper()}")
    return lines

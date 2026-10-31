"""'What changed since last run': pure diffs over saved packs and macro cards."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import changes as CH  # noqa: E402


def _pack(price=100.0, fv=50.0, rsi=60.0, tilt="hold"):
    return {"fundamentals": {"price": price}, "valuation": {"fair_value": fv},
            "technicals": {"rsi14": rsi},
            "macro_regime": {"tilt": {"regime": "mixed", "tilt": tilt}}}


def test_diff_packs_lists_only_moved_fields_with_percent_change():
    out = CH.diff_packs(_pack(), _pack(price=110.0, rsi=60.2, tilt="wait"))
    assert out == ["price: 100.00 → 110.00 (+10.0%)", "macro tilt: mixed/hold → mixed/wait"]


def test_diff_packs_ignores_missing_fields():
    assert CH.diff_packs({}, _pack()) == []


def test_previous_ticker_pack_picks_newest_earlier_and_skips_current(tmp_path, monkeypatch):
    monkeypatch.setattr(CH, "PACKS", tmp_path)
    monkeypatch.setattr(CH, "REPORTS", tmp_path)
    for stem in ("AAPL-20260101-1000", "AAPL-20260201-1000", "AAPL-20260301-1000"):
        (tmp_path / f"{stem}.json").write_text(json.dumps({"pack": _pack(price=float(stem[9:11]))}))
    (tmp_path / "AAPL-20260201-1000.md").write_text(
        "**FINAL: AVOID** · conviction 70/100 · size 0.0% · horizon 12mo\nPrice at report: **1.0**\n")
    pack, stem, report = CH.previous_ticker_pack("AAPL", "AAPL-20260301-1000")
    assert stem == "AAPL-20260201-1000" and report.signal == "sell"
    assert CH.previous_ticker_pack("AAPL", "AAPL-20260101-1000") is None


def test_diff_macro_reports_label_flips_and_tilt_changes():
    def card(label, applicable=True):
        return {"angles": {"real_yields": {"applicable": applicable, "label": label}}}
    out = CH.diff_macro(card("real yields rising"), card("real yields falling"),
                        {"tech": {"tilt": "hold"}, "gold": {"tilt": "wait"}},
                        {"tech": {"tilt": "add"}, "gold": {"tilt": "wait"}})
    assert out == ["real_yields: real yields rising → real yields falling", "tech tilt: HOLD → ADD"]

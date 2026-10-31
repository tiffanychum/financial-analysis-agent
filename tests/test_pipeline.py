"""Tests for pipeline.py's pure JSON-shaping helpers (no network, no LLM).
build_pack/write_report are exercised via the live pipeline, not here."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline as P  # noqa: E402


def _valuation_with_fv():
    return {
        "fair_value": 48.16, "upside_pct": -84.7,
        "sensitivity": {"rate_minus_1pct": 51.8, "rate_plus_1pct": 45.32},
        "confidence_range": {"applicable": True, "value": 48.9, "details": {"p10": 43.89, "p90": 53.93}},
        "models": {"reoi": {"applicable": True, "value": 47.57, "details": {}}},
        "quality": {
            "buyback_value_check": {
                "applicable": True, "value": 264.06,
                "details": {"implied_avg_price": 264.06, "buyback_cash": 90.7e9,
                           "shares_reduced": 343.5e6, "fair_value": 48.16,
                           "premium_to_fair_value_pct": 448.3,
                           "flag_buybacks_above_fair_value": True},
            },
            "altman_z": {"applicable": True, "value": 11.93, "details": {"zone": "safe"}},
        },
    }


def test_redact_valuation_strips_fair_value_and_bands():
    v = _valuation_with_fv()
    redacted = P._redact_valuation_for_domain_seats(v)
    assert "fair_value" not in redacted
    assert "upside_pct" not in redacted
    assert "sensitivity" not in redacted
    assert "confidence_range" not in redacted
    # unrelated valuation/quality data must survive untouched
    assert redacted["models"]["reoi"]["value"] == 47.57
    assert redacted["quality"]["altman_z"]["value"] == 11.93


def test_redact_valuation_strips_buyback_vs_fair_value_flag_only():
    v = _valuation_with_fv()
    redacted = P._redact_valuation_for_domain_seats(v)
    bb_details = redacted["quality"]["buyback_value_check"]["details"]
    assert "fair_value" not in bb_details
    assert "premium_to_fair_value_pct" not in bb_details
    assert "flag_buybacks_above_fair_value" not in bb_details
    # the non-FV-derived buyback facts (price paid, cash, shares) still show
    assert bb_details["implied_avg_price"] == 264.06
    assert bb_details["buyback_cash"] == 90.7e9


def test_redact_valuation_leaves_original_dict_untouched():
    v = _valuation_with_fv()
    P._redact_valuation_for_domain_seats(v)
    assert v["fair_value"] == 48.16  # caller's dict (full-view pack) unaffected
    assert v["quality"]["buyback_value_check"]["details"]["fair_value"] == 48.16


def test_pack_for_llm_full_view_includes_fair_value():
    pack = {"ticker": "AAPL", "as_of": "now", "fundamentals": {"name": "Apple"},
           "macro": {}, "technicals": {}, "news": [], "valuation": _valuation_with_fv()}
    out = json.loads(P.pack_for_llm(pack))
    assert out["valuation"]["fair_value"] == 48.16


def test_pack_for_llm_redacted_view_omits_fair_value():
    pack = {"ticker": "AAPL", "as_of": "now", "fundamentals": {"name": "Apple"},
           "macro": {}, "technicals": {}, "news": [], "valuation": _valuation_with_fv()}
    out = json.loads(P.pack_for_llm(pack, redact_fair_value=True))
    assert "fair_value" not in out["valuation"]
    assert "flag_buybacks_above_fair_value" not in out["valuation"]["quality"]["buyback_value_check"]["details"]


class _FakeModel:
    id = "test-model/high"

    def to_json(self):
        return {"id": self.id, "params": []}


class _FakeClient:
    model = _FakeModel()


def test_save_pack_writes_json_with_model_and_version(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PACKS", tmp_path / "packs")
    pack = {"ticker": "AAPL", "as_of": "now", "fundamentals": {}, "valuation": {}}
    path = P.save_pack(pack, _FakeClient(), ts="20260901-1200")
    assert path == tmp_path / "packs" / "AAPL-20260901-1200.json"
    saved = json.loads(path.read_text())
    assert saved["pipeline_version"] == P.PIPELINE_VERSION
    assert saved["model"]["id"] == "test-model/high"
    assert saved["pack"]["ticker"] == "AAPL"


def test_anchor_percentiles_etf_regime_only_calls_price_percentile(monkeypatch):
    calls = []
    monkeypatch.setattr(P.AN, "etf_price_percentile",
                        lambda ticker: calls.append(ticker) or {"applicable": True, "value": 1.0})
    monkeypatch.setattr(P.AN, "ps_percentile", lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    valuation = {"regime": {"regime": "etf"}, "models": {}}
    out = P._anchor_percentiles("QQQ", {}, valuation, rf=0.04, erp=0.05)
    assert calls == ["QQQ"]
    assert out == {"price_percentile": {"applicable": True, "value": 1.0}}


def test_anchor_percentiles_equity_regime_passes_solved_implied_growth(monkeypatch):
    monkeypatch.setattr(P.AN, "ps_percentile", lambda ticker, f: {"applicable": False, "reason": "x"})
    monkeypatch.setattr(P.AN, "ev_fcf_percentile", lambda ticker, f: {"applicable": False, "reason": "x"})
    seen = {}

    def fake_growth(ticker, f, rf, erp, current_growth):
        seen["current_growth"] = current_growth
        return {"applicable": False, "reason": "x"}

    monkeypatch.setattr(P.AN, "growth_percentile", fake_growth)
    valuation = {"regime": {"regime": "mature-profitable"},
                "models": {"reverse_dcf": {"applicable": True, "details": {"implied_growth": 0.123}}}}
    P._anchor_percentiles("AAPL", {}, valuation, rf=0.04, erp=0.05)
    # implied growth is read from reverse_dcf's already-solved value, not
    # re-solved here - anchors.growth_percentile only ranks it historically.
    assert seen["current_growth"] == pytest.approx(0.123)


def test_anchor_percentiles_equity_regime_current_growth_none_when_reverse_dcf_inapplicable(monkeypatch):
    monkeypatch.setattr(P.AN, "ps_percentile", lambda ticker, f: {"applicable": False, "reason": "x"})
    monkeypatch.setattr(P.AN, "ev_fcf_percentile", lambda ticker, f: {"applicable": False, "reason": "x"})
    seen = {}
    monkeypatch.setattr(P.AN, "growth_percentile",
                        lambda ticker, f, rf, erp, current_growth: seen.setdefault("g", current_growth))
    valuation = {"regime": {"regime": "distressed"},
                "models": {"reverse_dcf": {"applicable": False, "reason": "n/a"}}}
    P._anchor_percentiles("X", {}, valuation, rf=0.04, erp=0.05)
    assert seen["g"] is None


def test_anchor_report_lines_formats_applicable_and_na_entries():
    anchors = {
        "ps_percentile": {"applicable": True, "value": 6.5, "percentile": 75.0,
                          "n": 8, "years_covered": 2.0, "source": "SEC XBRL"},
        "ev_fcf_percentile": {"applicable": False, "reason": "current FCF is missing or non-positive"},
    }
    lines = P._anchor_report_lines(anchors)
    assert any("P/S: 6.5" in ln and "75th percentile" in ln for ln in lines)
    assert any(ln == "- EV/FCF: n/a — current FCF is missing or non-positive" for ln in lines)


def test_anchor_report_lines_placeholder_when_no_anchors_computed():
    lines = P._anchor_report_lines({})
    assert lines == ["n/a — anchors not computed for this pack (pre-p2-percentiles report)."]


def test_base_rate_report_lines_formats_applicable_and_unfired_entries():
    base_rates = {
        "years_covered": 14.9,
        "signals": {
            "golden_cross": {"applicable": True, "n": 5, "hit_rate": 0.8,
                             "hit_rate_95ci": (0.4, 0.95), "avg_forward_return_pct": 12.3,
                             "trusted": True},
            "death_cross": {"applicable": False, "n_fired": 0,
                           "reason": "fired 0 times with a full forward window in the lookback"},
        },
    }
    lines = P._base_rate_report_lines(base_rates)
    assert any("golden cross" in ln and "80% hit rate" in ln and "n=5" in ln for ln in lines)
    assert any("death cross: never fired" in ln for ln in lines)


def test_base_rate_report_lines_flags_untrusted_small_samples():
    base_rates = {"years_covered": 15.0,
                  "signals": {"macd_bull_cross": {"applicable": True, "n": 1, "hit_rate": 1.0,
                                                   "hit_rate_95ci": (0.2, 1.0),
                                                   "avg_forward_return_pct": 5.0, "trusted": False}}}
    lines = P._base_rate_report_lines(base_rates)
    assert any("small sample, do not treat as proof" in ln for ln in lines)


def test_base_rate_report_lines_placeholder_when_no_base_rates_computed():
    assert P._base_rate_report_lines({}) == [
        "n/a — base rates not computed for this pack (pre-p2-base-rates report)."]


def test_portfolio_report_lines_shows_weights_targets_and_drift():
    pf = {"applicable": True, "total_value": 10000.0, "currency": "USD", "cash_pct": 10.0,
         "holdings_pct": {"AAPL": 30.0, "EQQQ.L": 60.0}, "concentration_hhi": 4600.0,
         "current_ticker_existing_pct": 30.0,
         "targets_pct": {"AAPL": 20.0, "EQQQ.L": 60.0, "cash": 10.0}, "unpriced_holdings": []}
    lines = P._portfolio_report_lines(pf, "AAPL")
    assert any("Existing AAPL weight" in ln and "30.0%" in ln for ln in lines)
    assert any("AAPL: 30.0%" in ln and "target 20.0%" in ln and "+10.0pp" in ln for ln in lines)
    assert any("EQQQ.L: 60.0%" in ln and "drift +0.0pp" in ln for ln in lines)
    assert any(ln.startswith("- cash: 10.0%") and "target 10.0%" in ln for ln in lines)


def test_portfolio_report_lines_flags_unpriced_holdings():
    pf = {"applicable": True, "total_value": 1000.0, "currency": "USD", "cash_pct": 0.0,
         "holdings_pct": {"AAPL": 100.0}, "concentration_hhi": 10000.0,
         "current_ticker_existing_pct": None, "targets_pct": {}, "unpriced_holdings": ["DEAD"]}
    lines = P._portfolio_report_lines(pf, "AAPL")
    assert any("Could not price" in ln and "DEAD" in ln for ln in lines)


def test_portfolio_report_lines_not_applicable():
    pf = {"applicable": False, "reason": "no priceable holdings and no cash"}
    assert P._portfolio_report_lines(pf, "AAPL") == ["n/a — no priceable holdings and no cash"]


def test_save_pack_filename_matches_report_filename_stem(tmp_path, monkeypatch):
    """Same <TICKER>-<ts> stem as the .md report - the pairing is a filename
    match, not a separate index that could drift out of sync."""
    monkeypatch.setattr(P, "PACKS", tmp_path / "packs")
    pack_path = P.save_pack({"ticker": "TSLA", "fundamentals": {}, "valuation": {}},
                            _FakeClient(), ts="20260901-1200")
    assert pack_path.stem == "TSLA-20260901-1200"


def test_base_rate_report_lines_name_the_forward_window():
    lines = P._base_rate_report_lines({"years_covered": 15.0, "horizon_months": 3, "signals": {
        "death_cross": {"applicable": False, "n_fired": 2, "reason": "x"}}})
    assert lines[0] == "Own 15.0y history, 3-month forward window:"
    assert "full 3mo forward window" in lines[1]


@pytest.mark.parametrize("argv,expected", [
    (["AAPL"], 12),
    (["AAPL", "--horizon-months", "6"], 6),
    (["MACRO-BACKTEST", "QQQ", "--horizon-months", "3", "--no-whatsapp"], 3),
])
def test_horizon_arg_defaults_to_twelve_and_accepts_override(argv, expected):
    assert P._horizon_arg(argv) == expected


@pytest.mark.parametrize("bad", ["0", "61", "six", ""])
def test_horizon_arg_rejects_out_of_range_or_non_numeric(bad, capsys):
    argv = ["AAPL", "--horizon-months"] + ([bad] if bad else [])
    with pytest.raises(SystemExit) as exc:
        P._horizon_arg(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("ticker,ok", [("GC=F", True), ("^VIX", True), ("EQQQ.L", True),
                                       ("BRK-B", True), ("AAPL;rm", False)])
def test_valid_ticker_allows_yahoo_symbol_characters(ticker, ok):
    assert P._valid_ticker(ticker) is ok


@pytest.mark.parametrize("argv,expected", [
    (["pipeline.py", "aapl"], ("ticker", "AAPL", 12)),
    (["pipeline.py", "AAPL", "--horizon-months", "6"], ("ticker", "AAPL", 6)),
    (["pipeline.py", "--horizon-months", "6", "AAPL"], ("ticker", "AAPL", 6)),
    (["pipeline.py", "macro-backtest", "gc=f", "--horizon-months", "3"], ("backtest", "GC=F", 3)),
])
def test_main_dispatches_with_horizon(monkeypatch, argv, expected):
    calls = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(P, "run_ticker", lambda t, notify, h: calls.append(("ticker", t, h)))
    monkeypatch.setattr(P, "run_macro_backtest", lambda t, h, notify: calls.append(("backtest", t, h)))
    P.main()
    assert calls == [expected]


def _backtest_window(regimes: dict) -> dict:
    return {"applicable": True, "start": "2006-01", "end": "2025-09",
            "baseline": {"n_months": 237, "hit_rate": 0.82, "hit_rate_95ci": (0.77, 0.86),
                         "avg_return_pct": 15.1},
            "regimes": regimes}


def test_backtest_window_lines_show_baseline_and_flag_thin_regimes():
    lines = P._backtest_window_lines("Full window", _backtest_window({
        "headwind": {"n_months": 48, "n_episodes": 3, "hit_rate": 0.67, "hit_rate_95ci": (0.53, 0.79),
                     "avg_return_pct": 5.8, "trusted": False, "vs_baseline_return_pp": -9.3,
                     "vs_baseline_hit_rate_pp": -15.0}}))
    table = "\n".join(lines)
    assert "all months (baseline)** | 237" in table
    assert "headwind ⚠️ too few periods | 48 | 3 |" in table
    assert "-9.3pp return" in table


def test_backtest_window_lines_not_applicable():
    lines = P._backtest_window_lines("Recent window", {"applicable": False, "reason": "no data"})
    assert "n/a — no data" in lines


def test_send_whatsapp_is_noop_without_target(monkeypatch):
    import subprocess
    monkeypatch.setattr(P, "WHATSAPP_TO", "")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not shell out"))
    P.send_whatsapp("hi")


@pytest.mark.parametrize("ticker", [None, "AAPL"])
def test_run_track_writes_report_and_summary(ticker, tmp_path, monkeypatch, capsys):
    result = {"n_reports": 2, "hit_rate_by_horizon": {}, "macro_hit_rate_by_horizon": {},
              "invalidation": {"checked": 0, "triggered": 0, "complete": 0},
              "not_checkable": 0, "paper_books": {}}
    monkeypatch.setattr(P.TA, "run_tracker", lambda ticker=None: result)
    monkeypatch.setattr(P, "REPORTS", tmp_path)
    P.run_track(ticker, notify=False)
    assert "Track record" in capsys.readouterr().out
    (report,) = list(tmp_path.glob("TRACK-*.md"))
    assert (ticker or "TRACK") in report.name


def test_run_track_unknown_ticker_fails_cleanly(monkeypatch):
    msgs = []
    monkeypatch.setattr(P.TA, "run_tracker", lambda ticker=None: {"n_reports": 0})
    monkeypatch.setattr(P, "_fail", lambda msg, notify: msgs.append(msg))
    P.run_track("ZZZZ", notify=False)
    assert msgs and "no saved reports for ZZZZ" in msgs[0]

"""Tier 3 accuracy tracker: filename/markdown parsing, hit-rule logic, and the
idempotent SQLite scoring loop, all offline (no network)."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import track_accuracy as T  # noqa: E402


def _fake_history_df(close: float) -> pd.DataFrame:
    return pd.DataFrame({"Close": [close]})

REPORT_BODY = (
    "# AAPL — Apple — 2020-01-01\n\n"
    "**FINAL: BUY** · conviction 70/100 · size 5% · horizon 12mo\n"
    "Invalidation: x\n\n"
    "## Regime: mature-profitable — positive NOPAT\n"
    "Price at report: **100.00** USD\n\n"
    "Weighted ensemble (context, not the headline): **150.00** vs price 100.00 "
    "(upside 50.0%) · ±1% rate band [140 — 160]\n"
)

# ETF/ETC reports print "Price at report" unconditionally but skip the
# "Weighted ensemble" line entirely (no applicable model) - this is the
# exact shape that used to be unscoreable (EQQQ.L, IGLN.L).
ETF_REPORT_BODY = (
    "# EQQQ.L — Invesco EQQQ — 2020-01-01\n\n"
    "**FINAL: AVOID** · conviction 52/100 · size 0% · horizon 12mo\n"
    "Invalidation: x\n\n"
    "## Regime: etf — fund/ETF: no firm fundamentals\n"
    "Price at report: **530.50** GBP\n"
)


def _write_report(dir_: Path, name: str, body: str = REPORT_BODY) -> Path:
    p = dir_ / name
    p.write_text(body)
    return p


def test_parse_report_extracts_signal_conviction_price(tmp_path):
    p = _write_report(tmp_path, "AAPL-20200101-0900.md")
    got = T.parse_report(p)
    assert got.ticker == "AAPL"
    assert got.signal == "buy"
    assert got.conviction == 70
    assert got.price_at_report == 100.00
    assert got.report_date == datetime(2020, 1, 1, 9, 0, tzinfo=timezone.utc)


def test_parse_report_returns_none_for_unrecognized_filename(tmp_path):
    p = _write_report(tmp_path, "not-a-report.md")
    assert T.parse_report(p) is None


def test_parse_report_normalizes_avoid_label_to_sell_signal(tmp_path):
    # AVOID is the long-only display label pipeline.write_report prints for
    # what track_accuracy has always scored internally as "sell".
    p = _write_report(tmp_path, "EQQQ.L-20200101-0900.md", body=ETF_REPORT_BODY)
    got = T.parse_report(p)
    assert got.signal == "sell"
    assert got.price_at_report == 530.50


def test_parse_report_extracts_price_from_etf_report_with_no_ensemble_line(tmp_path):
    # The bug this fixes: a report with no applicable valuation model used
    # to have no parseable price at all, so it was silently unscoreable.
    p = _write_report(tmp_path, "EQQQ.L-20200101-0900.md", body=ETF_REPORT_BODY)
    got = T.parse_report(p)
    assert got.price_at_report is not None


def test_hit_rule_buy_sell_hold():
    assert T._hit("buy", 5.0) is True
    assert T._hit("buy", -1.0) is False
    assert T._hit("sell", -5.0) is True
    assert T._hit("sell", 1.0) is False
    assert T._hit("hold", 3.0) is True
    assert T._hit("hold", 50.0) is False


def test_score_report_none_when_too_recent(monkeypatch):
    report = T.ParsedReport(path=Path("x"), ticker="AAPL",
                            report_date=datetime.now(timezone.utc), signal="buy",
                            conviction=70, price_at_report=100.0)
    assert T.score_report(report, horizon_days=30) is None


def test_score_report_computes_return_and_hit(monkeypatch):
    old_date = datetime.now(timezone.utc) - timedelta(days=40)
    report = T.ParsedReport(path=Path("x"), ticker="AAPL", report_date=old_date,
                            signal="buy", conviction=70, price_at_report=100.0)

    class FakeTicker:
        def history(self, **kw):
            return _fake_history_df(110.0)

    monkeypatch.setattr(T.D, "_ticker", lambda ticker: FakeTicker())
    got = T.score_report(report, horizon_days=30)
    assert got["return_pct"] == 10.0
    assert got["hit"] is True


def test_score_report_computes_benchmark_relative_excess_return(monkeypatch):
    """Ticker up 10%, benchmark up 20% over the same window -> the BUY
    underperformed the benchmark even though it made money absolutely."""
    old_date = datetime.now(timezone.utc) - timedelta(days=40)
    report = T.ParsedReport(path=Path("x"), ticker="AAPL", report_date=old_date,
                            signal="buy", conviction=70, price_at_report=100.0)

    def fake_ticker(symbol: str):
        def history(**kw):
            if symbol == "^GSPC":
                # "start" kwarg -> the report-date lookup (_price_near);
                # otherwise -> the current-price lookup (_latest_price).
                return _fake_history_df(100.0 if "start" in kw else 120.0)
            return _fake_history_df(110.0)  # AAPL's current price
        return type("FT", (), {"history": staticmethod(history)})()

    monkeypatch.setattr(T.D, "_ticker", fake_ticker)
    got = T.score_report(report, horizon_days=30)
    assert got["benchmark_ticker"] == "^GSPC"
    assert got["return_pct"] == 10.0
    assert got["benchmark_return_pct"] == pytest.approx(20.0)
    assert got["excess_return_pct"] == pytest.approx(-10.0)
    assert got["hit"] is True            # absolute: made money on a buy
    assert got["hit_vs_benchmark"] is False  # relative: lagged the benchmark


def test_score_report_degrades_gracefully_when_benchmark_fetch_fails(monkeypatch):
    def fake_ticker(symbol: str):
        if symbol == "^GSPC":
            raise RuntimeError("network down")
        return type("FT", (), {"history": staticmethod(lambda **kw: _fake_history_df(110.0))})()

    monkeypatch.setattr(T.D, "_ticker", fake_ticker)
    old_date = datetime.now(timezone.utc) - timedelta(days=40)
    report = T.ParsedReport(path=Path("x"), ticker="AAPL", report_date=old_date,
                            signal="buy", conviction=70, price_at_report=100.0)
    got = T.score_report(report, horizon_days=30)
    assert got["return_pct"] == 10.0  # primary result unaffected
    assert got["benchmark_return_pct"] is None
    assert got["hit_vs_benchmark"] is None


def test_wilson_interval_one_of_one_is_wide_not_overconfident():
    # a raw hit_rate of 1/1 reads as "100%"; Wilson must not claim that
    lo, hi = T.wilson_interval(1, 1)
    assert lo == pytest.approx(0.2065, abs=1e-3)
    assert hi == 1.0


def test_wilson_interval_narrows_with_more_samples():
    lo, hi = T.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert hi - lo < 0.99  # much tighter than the n=1 case


def test_wilson_interval_none_for_zero_n():
    assert T.wilson_interval(0, 0) is None


def test_run_tracker_is_idempotent(tmp_path, monkeypatch):
    old_date = datetime.now(timezone.utc) - timedelta(days=400)
    fname = f"AAPL-{old_date.strftime('%Y%m%d-%H%M')}.md"
    _write_report(tmp_path, fname)
    monkeypatch.setattr(T, "DB_PATH", tmp_path / "track.sqlite")

    class FakeTicker:
        def history(self, **kw):
            return _fake_history_df(120.0)

    monkeypatch.setattr(T.D, "_ticker", lambda ticker: FakeTicker())
    first = T.run_tracker(reports_dir=tmp_path, horizons_days=(30, 90))
    assert first["scored_new"] == 2  # both horizons score on first pass
    second = T.run_tracker(reports_dir=tmp_path, horizons_days=(30, 90))
    assert second["scored_new"] == 0  # already scored -> no duplicate rows
    assert second["hit_rate_by_horizon"]["30d"]["n"] == 1


# ---------------------------------------------------------------- macro report scoring

MACRO_REPORT_BODY = (
    "# Macro regime — 2026-07-01 12:00 UTC\n\n"
    "**Tech ETF tilt: ADD** (supportive) — 3 supportive, 0 headwind, 1 neutral of 4 resolved angles\n"
    "**Gold tilt: WAIT** (headwind) — 0 supportive, 3 headwind, 1 neutral of 4 resolved angles\n"
    "Macro timing is weak at a 12-month horizon.\n\n"
    "## Scorecard\n"
)


def test_parse_macro_report_extracts_both_tilts(tmp_path):
    p = _write_report(tmp_path, "MACRO-20260701-1200.md", body=MACRO_REPORT_BODY)
    got = T.parse_macro_report(p)
    assert got.tilts == {"tech": "add", "gold": "wait"}
    assert got.report_date == datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def test_parse_macro_report_none_for_non_macro_filename(tmp_path):
    p = _write_report(tmp_path, "AAPL-20260701-1200.md", body=MACRO_REPORT_BODY)
    assert T.parse_macro_report(p) is None


def test_score_macro_tilt_backfills_price_near_report_date_no_printed_price(monkeypatch):
    # Unlike a ticker report, MACRO-*.md never prints a price line (see
    # write_macro_report) - score_macro_tilt must backfill it from the
    # proxy ticker's own history near report_date instead.
    old_date = datetime.now(timezone.utc) - timedelta(days=40)

    def fake_ticker(symbol: str):
        assert symbol == "QQQ"  # tech proxy from MACRO_PROXY_TICKERS
        return type("FT", (), {"history": staticmethod(
            lambda **kw: _fake_history_df(100.0 if "start" in kw else 110.0))})()

    monkeypatch.setattr(T.D, "_ticker", fake_ticker)
    got = T.score_macro_tilt("tech", "add", old_date, horizon_days=30)
    assert got["ticker"] == "QQQ"
    assert got["return_pct"] == pytest.approx(10.0)
    assert got["hit"] is True  # "add" -> buy semantics -> hit on positive return


def test_score_macro_tilt_wait_hits_on_negative_return(monkeypatch):
    old_date = datetime.now(timezone.utc) - timedelta(days=40)
    monkeypatch.setattr(T.D, "_ticker", lambda symbol: type("FT", (), {"history": staticmethod(
        lambda **kw: _fake_history_df(100.0 if "start" in kw else 90.0))})())
    got = T.score_macro_tilt("gold", "wait", old_date, horizon_days=30)
    assert got["signal"] == "sell"
    assert got["hit"] is True


def test_score_macro_tilt_none_when_too_recent():
    assert T.score_macro_tilt("tech", "add", datetime.now(timezone.utc), horizon_days=30) is None


def test_run_tracker_scores_macro_report_separately_from_ticker_reports(tmp_path, monkeypatch):
    old_date = datetime.now(timezone.utc) - timedelta(days=400)
    macro_fname = f"MACRO-{old_date.strftime('%Y%m%d-%H%M')}.md"
    _write_report(tmp_path, macro_fname, body=MACRO_REPORT_BODY)
    ticker_fname = f"AAPL-{old_date.strftime('%Y%m%d-%H%M')}.md"
    _write_report(tmp_path, ticker_fname)
    monkeypatch.setattr(T, "DB_PATH", tmp_path / "track.sqlite")
    monkeypatch.setattr(T.D, "_ticker", lambda ticker: type("FT", (), {"history": staticmethod(
        lambda **kw: _fake_history_df(120.0))})())

    result = T.run_tracker(reports_dir=tmp_path, horizons_days=(30, 90))
    # AAPL (1) + MACRO tech + MACRO gold (2), each at both horizons.
    assert result["scored_new"] == 6
    # The ticker hit rate must never include the macro tilt rows.
    assert result["hit_rate_by_horizon"]["30d"]["n"] == 1
    assert result["macro_hit_rate_by_horizon"]["30d"]["tech"]["n"] == 1
    assert result["macro_hit_rate_by_horizon"]["30d"]["gold"]["n"] == 1

    second = T.run_tracker(reports_dir=tmp_path, horizons_days=(30, 90))
    assert second["scored_new"] == 0  # idempotent, same as ticker-only reports


# ------------------------------------------------ Phase 3: invalidation + paper book

import pandas as pd  # noqa: E402
import sqlite3  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import track_accuracy as TA  # noqa: E402


def _closes(step_on="2024-06-30"):
    idx = pd.bdate_range("2024-01-01", "2026-10-20")
    return pd.Series([100.0 if d < pd.Timestamp(step_on) else 200.0 for d in idx], index=idx)


def test_parse_report_reads_size_horizon_and_invalidation_level(tmp_path):
    f = tmp_path / "AAPL-20260101-1200.md"
    f.write_text("**FINAL: AVOID** · conviction 78/100 · size 0.0% · horizon 6mo\n"
                 "Invalidation level: close below 294.00\nPrice at report: **347.73**\n")
    r = TA.parse_report(f)
    assert (r.size_pct, r.horizon_months, r.invalidation) == (0.0, 6, ("below", 294.0))


def test_parse_report_without_level_is_not_checkable(tmp_path):
    f = tmp_path / "AAPL-20260101-1200.md"
    f.write_text("**FINAL: HOLD** · conviction 50/100 · size 0.0% · horizon 12mo\nPrice at report: **10.0**\n")
    assert TA.parse_report(f).invalidation is None


def test_first_breach_is_strict_and_directional():
    s = pd.Series([10, 9, 8, 12], index=pd.bdate_range("2026-01-01", periods=4))
    assert TA.first_breach(s, "below", 9) == s.index[2]
    assert TA.first_breach(s, "above", 11) == s.index[3]
    assert TA.first_breach(s, "below", 5) is None


def test_check_invalidation_records_trigger_inside_window_only(monkeypatch, tmp_path):
    conn = sqlite3.connect(tmp_path / "t.sqlite")
    TA._ensure_schema(conn)
    monkeypatch.setattr(TA, "_closes_since", lambda t, d: _closes())
    report = TA.ParsedReport(path=tmp_path / "X.md", ticker="X",
                             report_date=datetime(2024, 1, 15, tzinfo=timezone.utc), signal="sell",
                             conviction=70, price_at_report=100.0, horizon_months=3,
                             invalidation=("above", 150.0))
    assert TA.check_invalidation(conn, report) is True
    row = conn.execute("SELECT triggered, window_complete FROM invalidation_checks").fetchone()
    assert row == (0, 1)  # the jump to 200 is after the 3-month window
    report.horizon_months = 12
    conn.execute("DELETE FROM invalidation_checks")
    TA.check_invalidation(conn, report)
    assert conn.execute("SELECT triggered, triggered_on FROM invalidation_checks").fetchone()[0] == 1


def test_paper_book_beats_dca_when_the_buy_came_before_the_rise():
    d = datetime(2024, 1, 15, tzinfo=timezone.utc)
    out = TA.simulate_books([(d, "buy", 10.0, 100.0)], _closes(), 200.0)
    assert round(out["agent_return_pct"]) == 100
    assert round(out["dca_return_pct"]) == 50


def test_paper_book_avoid_sells_everything_and_hold_does_nothing():
    d1, d2, d3 = (datetime(2024, m, 15, tzinfo=timezone.utc) for m in (1, 2, 3))
    out = TA.simulate_books([(d1, "buy", 5.0, 100.0), (d2, "hold", 0.0, 120.0), (d3, "sell", 0.0, 150.0)],
                            _closes(), 200.0)
    assert round(out["agent_return_pct"]) == 25  # half the cash 100->150 = +50% on 50%

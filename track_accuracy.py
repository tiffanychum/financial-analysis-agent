"""Tier 3: turn 'research only' into a measured track record.

Every `analyze <TICKER>` run already writes a dated report with a BUY/AVOID/
HOLD signal, a conviction score, and the price at report time. This script
revisits past reports, fetches the current price AND the price of a matched
benchmark over the same window, and scores whether price action agrees with
the signal both in absolute terms and relative to the benchmark — a BUY that
merely tracked a rising market is a weaker result than one that beat it, and
absolute scoring alone cannot tell the two apart.

Usage: .venv/bin/python track_accuracy.py [--min-age-days 30]

Storage: a small local SQLite db (reports/track_accuracy.sqlite), not a repo-
wide state store — this tool only tracks its own historical scoring runs, one
row per (report, days-since) pair, so re-running never double-counts.
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import data as D

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
DB_PATH = REPORTS / "track_accuracy.sqlite"

FILENAME_RE = re.compile(r"^([A-Za-z0-9.\-]+)-(\d{8})-(\d{4})\.md$")
# AVOID is the long-only display label for what was internally a "sell" call
# (pipeline.ACTION_LABEL) - older reports still literally say SELL, so both
# forms are accepted and normalized to "sell" right after matching.
SIGNAL_RE = re.compile(r"\*\*FINAL:\s*(BUY|SELL|AVOID|HOLD)\*\*.*?conviction (\d+)/100")
# "Price at report" is printed unconditionally (see pipeline.write_report),
# unlike the old "vs price" ensemble line which only existed when a
# valuation model applied - that gap is why ETF/ETC reports (EQQQ.L, IGLN.L)
# were never scorable at all.
PRICE_RE = re.compile(r"Price at report:\s*\*\*([\d.,]+)\*\*")
SIZE_RE = re.compile(r"conviction \d+/100 · size ([\d.]+)%")
HORIZON_RE = re.compile(r"· horizon (\d+)mo")
# Written by pipeline._invalidation_level_lines; absent on older reports,
# which only have a free-text invalidation and so are "not checkable".
INVALIDATION_RE = re.compile(r"^Invalidation level: close (below|above) ([\d.]+)", re.M)

# A HOLD only "wins" if price stayed within this band; BUY/SELL win on
# direction alone, matching the signal semantics used in short_summary().
HOLD_BAND_PCT = 10.0


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial hit rate. A raw hit_rate (or
    a normal-approximation interval) is wildly overconfident at the small n
    this tracker will have for a long time - 1/1 reads as "100%" with no
    hint that it is one coin flip. Wilson holds up at small n, unlike Wald."""
    if n == 0:
        return None
    p = hits / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


_BENCHMARK_COLUMNS = (
    ("benchmark_ticker", "TEXT"), ("benchmark_return_pct", "REAL"),
    ("excess_return_pct", "REAL"), ("hit_vs_benchmark", "INTEGER"),
)
# NULL for every normal ticker-analysis row; "tech"/"gold" for a row scoring
# the standalone `macro` report's tilt instead of a committee signal. Kept
# in the same table (same price-fetch/Wilson-interval machinery) but
# excluded from summarize()'s main hit-rate so a macro timing call and a
# stock-picking call never get silently averaged into one number.
_MACRO_COLUMNS = (("macro_asset_class", "TEXT"),)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scored_reports (
            report_path TEXT NOT NULL,
            ticker TEXT NOT NULL,
            report_date TEXT NOT NULL,
            signal TEXT NOT NULL,
            conviction INTEGER,
            price_at_report REAL,
            days_held INTEGER NOT NULL,
            price_now REAL,
            return_pct REAL,
            hit INTEGER,
            scored_at TEXT NOT NULL,
            PRIMARY KEY (report_path, days_held)
        )
    """)
    # Additive columns for benchmark-relative scoring - ALTER TABLE ADD
    # COLUMN on an existing db from before this feature, so a live
    # track_accuracy.sqlite doesn't need to be deleted to pick it up.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(scored_reports)")}
    for name, coltype in _BENCHMARK_COLUMNS + _MACRO_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE scored_reports ADD COLUMN {name} {coltype}")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS invalidation_checks (
            report_path TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            level REAL NOT NULL,
            window_end TEXT NOT NULL,
            window_complete INTEGER NOT NULL,
            triggered INTEGER NOT NULL,
            triggered_on TEXT,
            checked_at TEXT NOT NULL
        )
    """)
    conn.commit()


@dataclass
class ParsedReport:
    path: Path
    ticker: str
    report_date: datetime
    signal: str
    conviction: int | None
    price_at_report: float | None
    size_pct: float | None = None
    horizon_months: int = 12
    invalidation: tuple[str, float] | None = None


def parse_report(path: Path) -> ParsedReport | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    ticker, ymd, hm = m.groups()
    report_date = datetime.strptime(ymd + hm, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    text = path.read_text(errors="replace")
    sm = SIGNAL_RE.search(text)
    if not sm:
        return None
    pm = PRICE_RE.search(text)
    price = float(pm.group(1).replace(",", "")) if pm else None
    signal = sm.group(1).lower()
    if signal == "avoid":
        signal = "sell"
    zm, hm_ = SIZE_RE.search(text), HORIZON_RE.search(text)
    im = INVALIDATION_RE.search(text)
    return ParsedReport(path=path, ticker=ticker, report_date=report_date,
                        signal=signal, conviction=int(sm.group(2)),
                        price_at_report=price,
                        size_pct=float(zm.group(1)) if zm else None,
                        horizon_months=int(hm_.group(1)) if hm_ else 12,
                        invalidation=(im.group(1), float(im.group(2))) if im else None)


def _hit(signal: str, return_pct: float) -> bool:
    if signal == "buy":
        return return_pct > 0
    if signal == "sell":
        return return_pct < 0
    return abs(return_pct) <= HOLD_BAND_PCT  # hold


def _latest_price(ticker: str) -> float | None:
    try:
        hist = D._ticker(ticker).history(period="5d", auto_adjust=True)
        return float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
    except Exception:  # noqa: BLE001 - best-effort
        return None


def _price_near(ticker: str, date: datetime) -> float | None:
    """First close on/after `date` - a +/-6 day window covers weekends and a
    short holiday run without walking day-by-day."""
    try:
        hist = D._ticker(ticker).history(start=date.strftime("%Y-%m-%d"),
                                         end=(date + timedelta(days=6)).strftime("%Y-%m-%d"),
                                         auto_adjust=True)
        return float(hist["Close"].iloc[0]) if hist is not None and not hist.empty else None
    except Exception:  # noqa: BLE001 - best-effort
        return None


def score_report(report: ParsedReport, horizon_days: int) -> dict | None:
    if report.price_at_report is None:
        return None
    age_days = (datetime.now(timezone.utc) - report.report_date).days
    if age_days < horizon_days:
        return None  # too soon to score at this horizon
    price_now = _latest_price(report.ticker)
    if price_now is None:
        return None
    return_pct = (price_now - report.price_at_report) / report.price_at_report * 100

    # Benchmark-relative: a BUY that merely tracked a rising market is a
    # weaker result than one that beat it, and absolute-return scoring alone
    # cannot distinguish the two. Best-effort - a benchmark fetch failure
    # degrades to absolute-only scoring, it never blocks the primary result.
    bench = D.benchmark_for(report.ticker)
    bench_then = _price_near(bench, report.report_date)
    bench_now = _latest_price(bench)
    benchmark_return_pct = ((bench_now / bench_then - 1) * 100
                            if bench_then and bench_now else None)
    excess_return_pct = (return_pct - benchmark_return_pct
                        if benchmark_return_pct is not None else None)

    return {"price_now": price_now, "return_pct": return_pct,
            "hit": _hit(report.signal, return_pct),
            "benchmark_ticker": bench, "benchmark_return_pct": benchmark_return_pct,
            "excess_return_pct": excess_return_pct,
            "hit_vs_benchmark": (_hit(report.signal, excess_return_pct)
                                if excess_return_pct is not None else None)}


# Same tech-ETF/gold proxy tickers macro_rules.py's tilts are computed
# against - scoring always uses these two, regardless of which specific ETF
# a user separately analyzes, so the macro layer's own track record stays
# comparable day to day instead of drifting with ticker choice.
MACRO_PROXY_TICKERS = {"tech": "QQQ", "gold": "GC=F"}
TILT_TO_SIGNAL = {"add": "buy", "hold": "hold", "wait": "sell"}
MACRO_FILENAME_RE = re.compile(r"^MACRO-(\d{8})-(\d{4})\.md$")
TILT_RE = re.compile(r"\*\*(Tech ETF|Gold) tilt:\s*(ADD|HOLD|WAIT)\*\*")
_TILT_LABEL_TO_CLASS = {"Tech ETF": "tech", "Gold": "gold"}


@dataclass
class ParsedMacroReport:
    path: Path
    report_date: datetime
    tilts: dict[str, str]  # asset_class -> "add"/"hold"/"wait"


def parse_macro_report(path: Path) -> ParsedMacroReport | None:
    m = MACRO_FILENAME_RE.match(path.name)
    if not m:
        return None
    ymd, hm = m.groups()
    report_date = datetime.strptime(ymd + hm, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    text = path.read_text(errors="replace")
    tilts = {_TILT_LABEL_TO_CLASS[label]: tilt.lower() for label, tilt in TILT_RE.findall(text)}
    return ParsedMacroReport(path=path, report_date=report_date, tilts=tilts) if tilts else None


def score_macro_tilt(asset_class: str, tilt: str, report_date: datetime,
                     horizon_days: int) -> dict | None:
    """No printed price-at-report line to parse (see write_macro_report) -
    backfills the proxy's price near report_date the same way score_report
    backfills a benchmark's `bench_then`, so the tilt doesn't need its own
    price snapshot mechanism."""
    ticker = MACRO_PROXY_TICKERS[asset_class]
    price_then = _price_near(ticker, report_date)
    if price_then is None:
        return None
    age_days = (datetime.now(timezone.utc) - report_date).days
    if age_days < horizon_days:
        return None
    price_now = _latest_price(ticker)
    if price_now is None:
        return None
    return_pct = (price_now - price_then) / price_then * 100
    signal = TILT_TO_SIGNAL[tilt]
    return {"ticker": ticker, "price_at_report": price_then, "price_now": price_now,
            "return_pct": return_pct, "hit": _hit(signal, return_pct), "signal": signal}


def _score_macro_report(conn: sqlite3.Connection, path: Path,
                        horizons_days: tuple[int, ...]) -> int:
    parsed = parse_macro_report(path)
    if parsed is None:
        return 0
    scored = 0
    for asset_class, tilt in parsed.tilts.items():
        # report_path is per-asset-class (":tech"/":gold" suffix) - one
        # macro report yields two independent tilt predictions, and the
        # table's PRIMARY KEY is (report_path, days_held).
        report_path = f"{path}:{asset_class}"
        for horizon in horizons_days:
            if conn.execute("SELECT 1 FROM scored_reports WHERE report_path=? AND days_held=?",
                            (report_path, horizon)).fetchone():
                continue
            result = score_macro_tilt(asset_class, tilt, parsed.report_date, horizon)
            if result is None:
                continue
            conn.execute(
                "INSERT INTO scored_reports (report_path, ticker, report_date, signal, "
                "conviction, price_at_report, days_held, price_now, return_pct, hit, "
                "scored_at, macro_asset_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (report_path, result["ticker"], parsed.report_date.isoformat(), result["signal"],
                 None, result["price_at_report"], horizon, result["price_now"],
                 result["return_pct"], int(result["hit"]),
                 datetime.now(timezone.utc).isoformat(), asset_class))
            scored += 1
    conn.commit()
    return scored


def _naive(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert(None) if t.tzinfo else t


def _closes_since(ticker: str, start: datetime) -> pd.Series | None:
    """Daily closes from `start`, tz-naive index; None on any fetch failure."""
    try:
        hist = D._ticker(ticker).history(start=_naive(start).strftime("%Y-%m-%d"), auto_adjust=True)
    except Exception:  # noqa: BLE001 - best-effort
        return None
    if hist is None or hist.empty:
        return None
    close = hist["Close"].dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close


def first_breach(closes: pd.Series, direction: str, level: float) -> pd.Timestamp | None:
    """First daily close strictly below/above `level` (the judge's own stated
    invalidation), or None."""
    hit = closes[closes < level] if direction == "below" else closes[closes > level]
    return hit.index[0] if len(hit) else None


def check_invalidation(conn: sqlite3.Connection, report: ParsedReport) -> bool:
    """Did the report's stated invalidation level trigger within its holding
    period so far? A triggered row is final; an untriggered one is re-checked
    each run until its window is complete."""
    if report.invalidation is None:
        return False
    key = str(report.path)
    row = conn.execute("SELECT triggered, window_complete FROM invalidation_checks WHERE report_path=?",
                       (key,)).fetchone()
    if row and (row[0] or row[1]):
        return False
    direction, level = report.invalidation
    start = _naive(report.report_date)
    window_end = start + pd.DateOffset(months=report.horizon_months)
    closes = _closes_since(report.ticker, report.report_date)
    if closes is None:
        return False
    in_window = closes[closes.index <= window_end]
    hit = first_breach(in_window, direction, level)
    conn.execute(
        "INSERT OR REPLACE INTO invalidation_checks (report_path, ticker, direction, level, "
        "window_end, window_complete, triggered, triggered_on, checked_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (key, report.ticker, direction, level, window_end.isoformat(),
         int(pd.Timestamp.now() >= window_end), int(hit is not None),
         hit.isoformat() if hit is not None else None, datetime.now(timezone.utc).isoformat()))
    return True


def simulate_books(reports: list[tuple], closes: pd.Series, now_price: float,
                   start_cash: float = 10_000.0, dca_months: int = 12) -> dict:
    """Paper book following the reports vs a DCA book with identical cash.
    reports: [(date, signal, size_pct, price)] ascending. Agent: BUY invests
    size_pct/10 of the starting cash (10% = the product's max size = fully
    invested), AVOID sells everything, HOLD does nothing. DCA: 1/dca_months of
    the cash at monthly steps from the first report, whatever the reports
    said. Uninvested cash earns nothing in either book."""
    cash, shares = start_cash, 0.0
    for _, signal, size_pct, price in reports:
        if signal == "buy":
            spend = min(cash, start_cash * (size_pct or 0.0) / 10.0)
            shares += spend / price
            cash -= spend
        elif signal == "sell":
            cash += shares * price
            shares = 0.0
    agent_value = cash + shares * now_price

    first, now = _naive(reports[0][0]), pd.Timestamp.now()
    dca_cash, dca_shares = start_cash, 0.0
    for k in range(dca_months):
        when = first + pd.DateOffset(months=k)
        if when > now:
            break
        later = closes[closes.index >= when]
        if later.empty:
            continue
        dca_shares += (start_cash / dca_months) / float(later.iloc[0])
        dca_cash -= start_cash / dca_months
    dca_value = dca_cash + dca_shares * now_price
    return {"n_reports": len(reports), "n_buys": sum(r[1] == "buy" for r in reports),
            "first_report": first.date().isoformat(),
            "agent_return_pct": (agent_value / start_cash - 1) * 100,
            "dca_return_pct": (dca_value / start_cash - 1) * 100}


def run_paper_books(reports: list[ParsedReport]) -> dict[str, dict]:
    by_ticker: dict[str, list[ParsedReport]] = {}
    for r in reports:
        if r.price_at_report:
            by_ticker.setdefault(r.ticker, []).append(r)
    out = {}
    for ticker, rs in by_ticker.items():
        rs.sort(key=lambda r: r.report_date)
        closes = _closes_since(ticker, rs[0].report_date)
        if closes is None:
            continue
        out[ticker] = simulate_books([(r.report_date, r.signal, r.size_pct, r.price_at_report) for r in rs],
                                     closes, float(closes.iloc[-1]))
    return out


def run_tracker(reports_dir: Path = REPORTS, horizons_days: tuple[int, ...] = (30, 90, 180, 365)) -> dict:
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)
    scored, skipped = 0, 0
    for path in sorted(reports_dir.glob("*.md")):
        if path.name.startswith("MACRO-"):
            scored += _score_macro_report(conn, path, horizons_days)
            continue
        report = parse_report(path)
        if report is None:
            skipped += 1
            continue
        for horizon in horizons_days:
            existing = conn.execute(
                "SELECT 1 FROM scored_reports WHERE report_path=? AND days_held=?",
                (str(path), horizon)).fetchone()
            if existing:
                continue
            result = score_report(report, horizon)
            if result is None:
                continue
            conn.execute(
                "INSERT INTO scored_reports (report_path, ticker, report_date, signal, "
                "conviction, price_at_report, days_held, price_now, return_pct, hit, "
                "scored_at, benchmark_ticker, benchmark_return_pct, excess_return_pct, "
                "hit_vs_benchmark) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(path), report.ticker, report.report_date.isoformat(), report.signal,
                 report.conviction, report.price_at_report, horizon, result["price_now"],
                 result["return_pct"], int(result["hit"]),
                 datetime.now(timezone.utc).isoformat(), result["benchmark_ticker"],
                 result["benchmark_return_pct"], result["excess_return_pct"],
                 None if result["hit_vs_benchmark"] is None else int(result["hit_vs_benchmark"])))
            scored += 1
        conn.commit()
    parsed = [r for r in (parse_report(p) for p in sorted(reports_dir.glob("*.md"))
                          if not p.name.startswith("MACRO-")) if r]
    for r in parsed:
        check_invalidation(conn, r)
    conn.commit()
    summary = summarize(conn, horizons_days)
    summary["invalidation"] = dict(zip(
        ("checked", "triggered", "complete"),
        conn.execute("SELECT COUNT(*), COALESCE(SUM(triggered),0), COALESCE(SUM(window_complete),0) "
                     "FROM invalidation_checks").fetchone()))
    summary["not_checkable"] = sum(r.invalidation is None for r in parsed)
    conn.close()
    return {"scored_new": scored, "skipped_unparseable": skipped,
            "paper_books": run_paper_books(parsed), **summary}


def summarize(conn: sqlite3.Connection, horizons_days: tuple[int, ...]) -> dict:
    out = {}
    macro_out = {}
    for horizon in horizons_days:
        # macro_asset_class IS NULL excludes the macro report's tech/gold
        # tilt rows - macro timing calls and stock-picking calls are scored
        # from the same table but must never be averaged into one hit rate.
        rows = conn.execute(
            "SELECT hit, hit_vs_benchmark FROM scored_reports "
            "WHERE days_held=? AND macro_asset_class IS NULL", (horizon,)).fetchall()
        if rows:
            n = len(rows)
            hits = sum(r[0] for r in rows)
            stats = {"n": n, "hit_rate": hits / n, "hit_rate_95ci": wilson_interval(hits, n)}
            bench_rows = [r[1] for r in rows if r[1] is not None]
            if bench_rows:
                bn, bhits = len(bench_rows), sum(bench_rows)
                stats["hit_rate_vs_benchmark"] = bhits / bn
                stats["hit_rate_vs_benchmark_95ci"] = wilson_interval(bhits, bn)
                stats["n_vs_benchmark"] = bn
            out[f"{horizon}d"] = stats
        macro_rows = conn.execute(
            "SELECT macro_asset_class, hit FROM scored_reports "
            "WHERE days_held=? AND macro_asset_class IS NOT NULL", (horizon,)).fetchall()
        if macro_rows:
            by_class: dict[str, list[int]] = {}
            for cls, hit in macro_rows:
                by_class.setdefault(cls, []).append(hit)
            macro_out[f"{horizon}d"] = {
                cls: {"n": len(hits), "hit_rate": sum(hits) / len(hits),
                     "hit_rate_95ci": wilson_interval(sum(hits), len(hits))}
                for cls, hits in by_class.items()}
    return {"hit_rate_by_horizon": out, "macro_hit_rate_by_horizon": macro_out}


def main() -> None:
    result = run_tracker()
    print(f"Scored {result['scored_new']} new (ticker, horizon) pairs; "
          f"{result['skipped_unparseable']} reports unparseable.")
    for horizon, stats in result["hit_rate_by_horizon"].items():
        lo, hi = stats["hit_rate_95ci"]
        line = f"  {horizon}: hit rate {stats['hit_rate']:.0%} [{lo:.0%}-{hi:.0%}] (n={stats['n']})"
        if "hit_rate_vs_benchmark" in stats:
            blo, bhi = stats["hit_rate_vs_benchmark_95ci"]
            line += (f" · vs benchmark {stats['hit_rate_vs_benchmark']:.0%} "
                    f"[{blo:.0%}-{bhi:.0%}] (n={stats['n_vs_benchmark']})")
        print(line)
    if not result["hit_rate_by_horizon"]:
        print("  No reports old enough yet to score at any horizon.")
    inv = result["invalidation"]
    print(f"Invalidation levels: {inv['checked']} checkable ({inv['complete']} windows complete), "
          f"{inv['triggered']} triggered; {result['not_checkable']} older reports have no checkable level.")
    for ticker, b in result["paper_books"].items():
        print(f"Paper book {ticker} (since {b['first_report']}, {b['n_reports']} reports, "
              f"{b['n_buys']} buys): agent {b['agent_return_pct']:+.1f}% vs DCA {b['dca_return_pct']:+.1f}%")
    if result["macro_hit_rate_by_horizon"]:
        print("Macro tilt hit rate (tech ETF/gold, scored vs QQQ/GC=F, separate from stock picks):")
        for horizon, by_class in result["macro_hit_rate_by_horizon"].items():
            for cls, stats in by_class.items():
                lo, hi = stats["hit_rate_95ci"]
                print(f"  {horizon} {cls}: hit rate {stats['hit_rate']:.0%} "
                     f"[{lo:.0%}-{hi:.0%}] (n={stats['n']})")


if __name__ == "__main__":
    sys.exit(main())

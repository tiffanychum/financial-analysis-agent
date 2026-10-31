"""Macro data layer: keyless fetchers for Fed policy, liquidity, inflation,
real yields, growth/recession risk, financial conditions, dollar/FX, the
FOMC statement pair (for the redline in macro_rules.py), and two historical
relationship checks (gold vs real yields, QQQ vs net liquidity).

Every fetch is isolated in its own try/except and never raises — a missing
series means None + a named reason downstream, the same "no silent zeros"
posture as data.py. Every value carries a `_source` (series id + as-of date)
so a report line is never mistaken for a guess.

Sources (all live-verified 2026-09-05, no API key required):
- FRED `fredgraph.csv` (official, weekly/daily/monthly series)
- federalreserve.gov FOMC calendar + statement pages (HTML scrape, stable
  URL pattern `monetaryYYYYMMDDa.htm`)
- Yahoo Finance via data.py's shared browser-impersonating session (ZQ=F
  fed funds futures, GC=F gold, ^VIX, GBPUSD=X, and QQQ for the liquidity
  relationship check)

Explicitly NOT built here: BLS CPI/NFP schedule pages 403 non-browser
requests (live-checked), and Yale Budget Lab's tariff page is a heavy
client-rendered app with no stable scrape target (live-checked) - both are
weak, low-payoff scrape targets. Trade/tariff commentary instead goes
through the macro LLM committee's fiscal/trade reader with webSearch,
tagged unverified, matching agents_llm.fetch_market_consensus's posture for
the same kind of gap.
"""

from __future__ import annotations

import csv
import io
import math
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

import data as D

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# WALCL and WTREGEN are published in millions of USD, RRPONTSYD in billions
# (live-checked: 2022-12-30 RRPONTSYD = 2553.7 = $2.55T). Net liquidity must
# rescale RRP first or it is overstated by up to ~$2.5T in 2021-2023.
RRP_TO_MILLIONS = 1000.0
FOMC_CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_STMT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{ymd}a.htm"


def _get(url: str, timeout: float = 15.0) -> str:
    """FRED and federalreserve.gov both silently stall plain urllib reads
    (live-checked: connects fine, then the read blocks past any timeout -
    the same bot-throttling class data.py already documents for Yahoo), but
    respond instantly to a real-browser TLS/HTTP fingerprint. Reuse data.py's
    shared curl_cffi session rather than adding a second HTTP client."""
    r = D._yf_session().get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------- FRED

def fred_series(series_id: str, years_back: int = 3) -> list[tuple[str, float]]:
    """(date, value) pairs, oldest first, most recent last. `cosd` (custom
    observation start date) caps the payload instead of pulling full history
    - WALCL/CPILFESL go back to the 1940s-60s for a metric we only ever look
    at 1-3y of. FRED prints "." for missing observations; those rows are
    dropped rather than coerced to 0."""
    cosd = (datetime.now(timezone.utc) - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    text = _get(f"{FRED_BASE}?id={series_id}&cosd={cosd}")
    rows = list(csv.reader(io.StringIO(text)))
    out = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        try:
            out.append((row[0], float(row[1])))
        except ValueError:
            continue
    return out


def _nearest_on_or_before(rows: list[tuple[str, float]], target: str) -> tuple[str, float] | None:
    for date, val in reversed(rows):
        if date <= target:
            return date, val
    return None


def fred_latest(series_id: str, years_back: int = 3) -> tuple[float | None, str | None, str | None]:
    """(value, date, source) for the most recent observation."""
    try:
        rows = fred_series(series_id, years_back)
    except Exception:  # noqa: BLE001 - best-effort; caller keeps field None
        return None, None, None
    if not rows:
        return None, None, None
    date, val = rows[-1]
    return val, date, f"FRED {series_id} ({date})"


def fred_change(series_id: str, days_back: int, years_back: int = 3
                ) -> tuple[float | None, float | None, str | None, str | None]:
    """(latest, then, latest_date, then_date) — `then` is the nearest
    observation on/before `days_back` days before the latest one. Nearest-
    on-or-before (not a fixed row-index offset) so this stays correct across
    daily, weekly, and monthly series with different gaps/holidays."""
    try:
        rows = fred_series(series_id, years_back)
    except Exception:  # noqa: BLE001
        return None, None, None, None
    if not rows:
        return None, None, None, None
    latest_date, latest_val = rows[-1]
    target = (datetime.strptime(latest_date, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
    then = _nearest_on_or_before(rows[:-1], target)
    if then is None:
        return latest_val, None, latest_date, None
    then_date, then_val = then
    return latest_val, then_val, latest_date, then_date


def fred_yoy_pct(series_id: str, years_back: int = 3) -> tuple[float | None, str | None]:
    latest, then, latest_date, _ = fred_change(series_id, 365, years_back)
    if latest is None or then is None or then == 0:
        return None, None
    return (latest / then - 1) * 100, latest_date


def fred_annualized_pct(series_id: str, months: int, years_back: int = 3) -> float | None:
    """Annualized % change over the trailing `months` — the "is the recent
    trend hotter or cooler than the YoY headline" read (e.g. 3m annualized
    core PCE vs its YoY)."""
    latest, then, _, _ = fred_change(series_id, months * 30, years_back)
    if latest is None or then is None or then <= 0:
        return None
    periods_per_year = 12 / months
    return ((latest / then) ** periods_per_year - 1) * 100


# ---------------------------------------------------------------- Yahoo (via data.py's shared session)

def yahoo_price(symbol: str) -> tuple[float | None, str | None]:
    try:
        fi = D._ticker(symbol).fast_info
        px = D._f(fi.get("lastPrice"))
        return (px, f"yfinance {symbol}") if px is not None else (None, None)
    except Exception:  # noqa: BLE001
        return None, None


def fed_funds_futures_implied_rate() -> tuple[float | None, str | None]:
    """Front-month 30-day Fed Funds future (CME `ZQ`): implied rate = 100 -
    price. This is a real traded price, not a survey — the "priced" half of
    "priced vs said" (the FOMC statement's target range is the "said" half,
    see macro_rules.policy_path)."""
    px, _ = yahoo_price("ZQ=F")
    if px is None:
        return None, None
    return 100.0 - px, "CME 30-Day Fed Funds futures (ZQ=F front month), 100-price"


# ---------------------------------------------------------------- FOMC calendar + statements

_MONTH_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:-(\d{1,2}))?\*?"
)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}


def _strip_html(html: str) -> str:
    clean = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", clean)
    text = text.replace("&#8211;", "-").replace("&ndash;", "-").replace("&nbsp;", " ")
    text = text.replace("&rsquo;", "'").replace("&#x27;", "'").replace("&#39;", "'")
    return re.sub(r"\s+", " ", text).strip()


def fomc_meeting_dates(year: int) -> list[tuple[datetime, bool]]:
    """(meeting_end_date, has_statement) for every FOMC meeting in `year`,
    scraped from the official calendar page. "has_statement" distinguishes
    already-held meetings (statement published) from scheduled-but-not-yet-
    held ones - the calendar lists the whole year's dates up front, so a
    date's own text alone can't tell held-vs-future; that's a separate flag
    the caller (next_fomc_meeting) needs to pick the right one."""
    html = _get(FOMC_CAL_URL)
    text = _strip_html(html)
    start = text.find(f"{year} FOMC Meetings")
    if start < 0:
        return []
    end = text.find(" FOMC Meetings", start + len(f"{year} FOMC Meetings"))
    block = text[start:end if end > 0 else start + 4000]
    out = []
    for m in _MONTH_RE.finditer(block):
        month, d1, d2 = m.group(1), int(m.group(2)), m.group(3)
        day = int(d2) if d2 else d1
        try:
            date = datetime(year, _MONTHS[month], day, tzinfo=timezone.utc)
        except ValueError:
            continue
        after = block[m.end():m.end() + 20]
        out.append((date, after.strip().startswith("Statement:")))
    return out


def next_fomc_meeting() -> tuple[datetime | None, str | None]:
    """(date, source) of the next scheduled (not-yet-held) FOMC meeting.
    Falls through to next year's calendar only if the current year's list
    is exhausted - a real but rare edge case (matters only in the last ~3
    weeks of December) not worth a second scrape in the common case."""
    now = datetime.now(timezone.utc)
    for year in (now.year, now.year + 1):
        try:
            dates = fomc_meeting_dates(year)
        except Exception:  # noqa: BLE001
            continue
        upcoming = sorted(d for d, has_stmt in dates if not has_stmt and d >= now)
        if upcoming:
            return upcoming[0], "federalreserve.gov FOMC calendar"
    return None, None


def fetch_fomc_statement(meeting_end_date: datetime) -> tuple[str | None, str | None]:
    """Full statement text (between "approved the following statement" and
    "For media inquiries") for the meeting ending on this date. Returns
    (text, url); either is None on failure."""
    ymd = meeting_end_date.strftime("%Y%m%d")
    url = FOMC_STMT_URL.format(ymd=ymd)
    try:
        text = _strip_html(_get(url))
    except Exception:  # noqa: BLE001
        return None, None
    start = text.find("approved the following statement")
    if start < 0:
        return None, url
    start = text.find(":", start) + 1
    end = text.find("For media inquiries", start)
    if start <= 0 or end < 0:
        return None, url
    return text[start:end].strip(), url


def latest_two_fomc_statements() -> dict:
    """The most recent held meeting's statement plus the one before it, for
    macro_rules.fomc_redline. Held meetings only (has_statement=True) -
    an unheld meeting has no statement page yet."""
    now = datetime.now(timezone.utc)
    dates: list[tuple[datetime, bool]] = []
    for year in (now.year, now.year - 1):
        try:
            dates += fomc_meeting_dates(year)
        except Exception:  # noqa: BLE001
            continue
    held = sorted((d for d, has_stmt in dates if has_stmt), reverse=True)
    if len(held) < 1:
        return {}
    latest_text, latest_url = fetch_fomc_statement(held[0])
    out = {"latest_date": held[0].strftime("%Y-%m-%d"), "latest_text": latest_text,
           "latest_url": latest_url, "source": "federalreserve.gov FOMC statement"}
    if len(held) >= 2:
        prev_text, prev_url = fetch_fomc_statement(held[1])
        out.update({"prev_date": held[1].strftime("%Y-%m-%d"), "prev_text": prev_text,
                    "prev_url": prev_url})
    return out


_TARGET_RANGE_RE = re.compile(
    r"target range for the federal funds rate at (\d+(?:-\d+/\d+)?) to (\d+(?:-\d+/\d+)?) percent")


def _mixed_fraction_to_float(s: str) -> float | None:
    """Parses Fed statement number formats: "3-1/2" (mixed fraction) or a
    plain integer/decimal string."""
    m = re.match(r"(\d+)-(\d+)/(\d+)$", s)
    if m:
        whole, num, den = (int(x) for x in m.groups())
        return whole + num / den
    try:
        return float(s)
    except ValueError:
        return None


def parse_target_range(statement_text: str) -> tuple[float, float] | None:
    m = _TARGET_RANGE_RE.search(statement_text or "")
    if not m:
        return None
    lo, hi = _mixed_fraction_to_float(m.group(1)), _mixed_fraction_to_float(m.group(2))
    return (lo, hi) if lo is not None and hi is not None else None


# ---------------------------------------------------------------- relationship checks

def _weekly_series_from_fred(series_id: str, years_back: int = 3) -> pd.Series:
    rows = fred_series(series_id, years_back)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(d): v for d, v in rows}).sort_index()
    return s.resample("W").last().ffill()


def _weekly_series_from_yahoo(symbol: str, years_back: int = 3) -> pd.Series:
    """Daily history resampled to week-ending-Sunday ("W"), matching
    _weekly_series_from_fred's grid exactly - Yahoo's own `interval="1wk"`
    bars land on whatever weekday the symbol's first bar happened to fall
    on (live-checked: Mondays for GC=F), which never lines up with FRED's
    Sunday-ending weeks and leaves every join empty after dropna()."""
    try:
        hist = D._ticker(symbol).history(period=f"{years_back}y", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return pd.Series(dtype=float)
        s = hist["Close"]
        s.index = s.index.tz_localize(None)
        return s.resample("W").last().ffill()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


def rolling_correlation(fred_id: str, yahoo_symbol: str, years_back: int = 3,
                        fred_is_level_diff: bool = True) -> tuple[float | None, int | None, str | None]:
    """Weekly-resampled correlation between a FRED series' change and a
    Yahoo instrument's return, over `years_back` years — "does the textbook
    relationship currently hold" (e.g. gold vs real yields should be
    negative; QQQ vs net liquidity should be positive), computed rather
    than assumed. fred_is_level_diff=True diffs the FRED series (real
    yields, a rate) instead of taking a % change (which breaks near/at 0)."""
    try:
        fred_s = _weekly_series_from_fred(fred_id, years_back)
        yahoo_s = _weekly_series_from_yahoo(yahoo_symbol, years_back)
        if fred_s.empty or yahoo_s.empty:
            return None, None, None
        fred_chg = fred_s.diff() if fred_is_level_diff else fred_s.pct_change()
        yahoo_ret = yahoo_s.pct_change()
        joined = pd.concat([fred_chg, yahoo_ret], axis=1, keys=["f", "y"]).dropna()
        if len(joined) < 26:
            return None, None, None
        corr = float(joined["f"].corr(joined["y"]))
        return (None if math.isnan(corr) else corr), len(joined), \
            f"weekly {fred_id} chg vs {yahoo_symbol} return, n={len(joined)}, {years_back}y"
    except Exception:  # noqa: BLE001
        return None, None, None


def net_liquidity_series(years_back: int = 1) -> pd.Series:
    """WALCL - WTREGEN - RRPONTSYD, aligned to WALCL's (weekly) dates —
    the QQQ-vs-liquidity relationship check needs a single aligned series,
    not three independently-cadenced ones."""
    walcl = _weekly_series_from_fred("WALCL", years_back)
    tga = _weekly_series_from_fred("WTREGEN", years_back)
    rrp = _weekly_series_from_fred("RRPONTSYD", years_back) * RRP_TO_MILLIONS
    joined = pd.concat([walcl, tga, rrp], axis=1, keys=["w", "t", "r"]).ffill().dropna()
    return joined["w"] - joined["t"] - joined["r"]


def net_liquidity_vs_qqq_correlation(years_back: int = 3) -> tuple[float | None, int | None, str | None]:
    try:
        nl = net_liquidity_series(years_back)
        qqq = _weekly_series_from_yahoo("QQQ", years_back)
        if nl.empty or qqq.empty:
            return None, None, None
        joined = pd.concat([nl.diff(), qqq.pct_change()], axis=1, keys=["nl", "q"]).dropna()
        if len(joined) < 26:
            return None, None, None
        corr = float(joined["nl"].corr(joined["q"]))
        return (None if math.isnan(corr) else corr), len(joined), \
            f"weekly net-liquidity chg vs QQQ return, n={len(joined)}, {years_back}y"
    except Exception:  # noqa: BLE001
        return None, None, None


# ---------------------------------------------------------------- assembly

def _series(field: str, series_id: str, years_back: int = 3, scale: float = 1.0) -> dict:
    val, date, src = fred_latest(series_id, years_back)
    if val is None:
        return {"value": None, "as_of": None, "source": f"FRED {series_id} unreachable"}
    if scale != 1.0:
        src += f", rescaled x{scale:g} to millions"
    return {"value": val * scale, "as_of": date, "source": src}


def build_macro_pack() -> dict:
    """Top-level: one dict with every angle, each field carrying `_source`
    or nested `source`, never raising. Standalone `macro` command and
    `analyze <TICKER>`'s ETF-regime layering both consume this same shape."""
    pack: dict = {"as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

    fomc = latest_two_fomc_statements()
    pack["fomc"] = fomc
    target_range = parse_target_range(fomc.get("latest_text") or "")
    implied_rate, implied_source = fed_funds_futures_implied_rate()
    pack["policy_path"] = {
        "target_range_low": target_range[0] if target_range else None,
        "target_range_high": target_range[1] if target_range else None,
        "target_range_source": fomc.get("latest_url"),
        "futures_implied_rate": implied_rate, "futures_implied_source": implied_source,
    }

    walcl = _series("fed_balance_sheet", "WALCL", years_back=1)
    reserves = _series("bank_reserves", "WRESBAL", years_back=1)
    rrp = _series("reverse_repo", "RRPONTSYD", years_back=1, scale=RRP_TO_MILLIONS)
    tga = _series("treasury_general_account", "WTREGEN", years_back=1)
    net_liq, net_liq_3m_chg_pct = None, None
    if all(x["value"] is not None for x in (walcl, tga, rrp)):
        net_liq = walcl["value"] - tga["value"] - rrp["value"]
        try:
            nl_series = net_liquidity_series(years_back=1)
            if len(nl_series) > 13:  # weekly series: ~13 obs back = 3 months
                net_liq_3m_chg_pct = (nl_series.iloc[-1] / nl_series.iloc[-13] - 1) * 100
        except Exception:  # noqa: BLE001 - best-effort; level above still stands
            pass
    pack["liquidity"] = {
        "fed_balance_sheet": walcl, "bank_reserves": reserves,
        "reverse_repo": rrp, "treasury_general_account": tga,
        "net_liquidity": net_liq, "net_liquidity_3m_change_pct": net_liq_3m_chg_pct,
        "net_liquidity_source": "WALCL - WTREGEN - RRPONTSYD (FRED, latest of each)" if net_liq is not None else None,
    }

    pcepilfe_yoy, pcepilfe_date = fred_yoy_pct("PCEPILFE")
    cpilfesl_yoy, cpilfesl_date = fred_yoy_pct("CPILFESL")
    pack["inflation"] = {
        "core_pce_yoy_pct": pcepilfe_yoy, "core_pce_3m_annualized_pct": fred_annualized_pct("PCEPILFE", 3),
        "core_pce_as_of": pcepilfe_date, "core_pce_source": "FRED PCEPILFE, YoY computed",
        "core_cpi_yoy_pct": cpilfesl_yoy, "core_cpi_3m_annualized_pct": fred_annualized_pct("CPILFESL", 3),
        "core_cpi_as_of": cpilfesl_date, "core_cpi_source": "FRED CPILFESL, YoY computed",
        "breakeven_10y": _series("breakeven_10y", "T10YIE", years_back=1),
        "breakeven_5y5y_forward": _series("breakeven_5y5y_forward", "T5YIFR", years_back=1),
    }

    dfii10 = _series("real_yield_10y", "DFII10", years_back=1)
    dfii10_latest, dfii10_then, _, dfii10_then_date = fred_change("DFII10", 90, years_back=1)
    pack["real_yields"] = {
        "real_10y": dfii10,
        "real_10y_3m_change_bps": (dfii10_latest - dfii10_then) * 100
        if dfii10_latest is not None and dfii10_then is not None else None,
        "real_10y_3m_change_from": dfii10_then_date,
    }

    pack["growth_recession"] = {
        "unemployment_rate": _series("unemployment_rate", "UNRATE", years_back=2),
        "sahm_rule": _series("sahm_rule_realtime", "SAHMREALTIME", years_back=2),
        "initial_claims": _series("initial_jobless_claims", "ICSA", years_back=1),
        "yield_curve_10y2y": _series("yield_curve_10y2y", "T10Y2Y", years_back=1),
    }

    vix, vix_src = yahoo_price("^VIX")
    pack["financial_conditions"] = {
        "high_yield_spread": _series("hy_credit_spread", "BAMLH0A0HYM2", years_back=1),
        "vix": {"value": vix, "source": vix_src},
    }

    dxy_broad = _series("broad_dollar_index", "DTWEXBGS", years_back=1)
    dxy_latest, dxy_then, _, dxy_then_date = fred_change("DTWEXBGS", 90, years_back=1)
    dxy_3m_chg_pct = ((dxy_latest / dxy_then - 1) * 100
                      if dxy_latest is not None and dxy_then else None)
    gbpusd, gbpusd_src = yahoo_price("GBPUSD=X")
    gold, gold_src = yahoo_price("GC=F")
    pack["dollar_fx"] = {
        "broad_dollar_index": dxy_broad,
        "broad_dollar_index_3m_change_pct": dxy_3m_chg_pct,
        "broad_dollar_index_3m_change_from": dxy_then_date,
        "gbpusd": {"value": gbpusd, "source": gbpusd_src},
    }
    pack["gold_price"] = {"value": gold, "source": gold_src}

    next_meeting, next_meeting_source = next_fomc_meeting()
    days_to_fomc = (next_meeting - datetime.now(timezone.utc)).days if next_meeting else None
    pack["event_calendar"] = {
        "next_fomc_date": next_meeting.strftime("%Y-%m-%d") if next_meeting else None,
        "next_fomc_days": days_to_fomc, "next_fomc_source": next_meeting_source,
        # BLS's own unbroken convention (not a scrape/guess) - the schedule
        # page itself 403s non-browser requests (live-checked), so this is
        # the strongest free signal available for the NFP date specifically.
        "next_nfp_note": "NFP release convention: first Friday of the month",
        "binary_event_within_3d": days_to_fomc is not None and 0 <= days_to_fomc <= 3,
    }

    gold_corr, gold_corr_n, gold_corr_src = rolling_correlation("DFII10", "GC=F", years_back=3)
    liq_corr, liq_corr_n, liq_corr_src = net_liquidity_vs_qqq_correlation(years_back=3)
    pack["relationship_checks"] = {
        "gold_vs_real_yield_corr": gold_corr, "gold_vs_real_yield_n": gold_corr_n,
        "gold_vs_real_yield_source": gold_corr_src,
        "qqq_vs_net_liquidity_corr": liq_corr, "qqq_vs_net_liquidity_n": liq_corr_n,
        "qqq_vs_net_liquidity_source": liq_corr_src,
    }
    return pack

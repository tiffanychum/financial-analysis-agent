"""Data layer: yfinance primary (simple, reliable), OpenBB used for extras
(options chain metrics, analyst estimates) with graceful skip. Produces the
plain fundamentals dict that valuation.py consumes, plus technicals and news
headlines. Headlines are the only untrusted text and are passed downstream as
data-only strings (prompt-injection posture: same as the bug-watcher).
"""

from __future__ import annotations

import csv
import io
import math
import re
import ssl
import urllib.request
from datetime import datetime, timedelta, timezone

import certifi
import pandas as pd
import yfinance as yf
from curl_cffi import requests as creq

RF_FALLBACK = 0.042  # used only when treasury.gov and ^TNX are both unreachable
ERP_FALLBACK = 0.05  # used only when Damodaran's page is unreachable/unparseable

_SESSION: creq.Session | None = None


def _yf_session() -> creq.Session:
    """A single shared browser-impersonating session for every yfinance call.
    Yahoo's default python-requests fingerprint gets bot-detected/rate-limited
    (401 'Invalid Crumb' or 429) on some networks even with a real crumb flow;
    curl_cffi's TLS/HTTP2 impersonation of a real Chrome client is yfinance's
    own documented workaround and fixes the chart/fundamentals/fast_info
    endpoints. quoteSummary (`.info`, growth_estimates, analyst targets) can
    still be blocked on IP-reputation-restricted networks regardless -
    fetch_fundamentals() has a fast_info fallback for exactly that case."""
    global _SESSION
    if _SESSION is None:
        _SESSION = creq.Session(impersonate="chrome")
    return _SESSION


def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol, session=_yf_session())


_SEC_UA = {"User-Agent": "openclaw-finance-agent (research tool; no-reply@openclaw.local)"}
_SEC_CIK_MAP: dict[str, int] | None = None


def _sec_cik_map() -> dict[str, int]:
    """SEC's free ticker->CIK mapping: no auth, no API key, official source.
    Cached once per process — the file is ~800KB and ticker->CIK barely
    changes, so re-fetching per ticker would be wasteful and rude to SEC."""
    global _SEC_CIK_MAP
    if _SEC_CIK_MAP is None:
        try:
            r = _yf_session().get("https://www.sec.gov/files/company_tickers.json",
                                  timeout=10, headers=_SEC_UA)
            _SEC_CIK_MAP = {v["ticker"]: v["cik_str"] for v in r.json().values()}
        except Exception:  # noqa: BLE001 - best-effort; empty map = every lookup skips
            _SEC_CIK_MAP = {}
    return _SEC_CIK_MAP


def _sic_sector(sic: int | None) -> str | None:
    """SIC division H (finance/insurance/real estate) is the only sector
    bucket valuation.classify_regime() actually branches on (equity-RIM
    routing, REIT exclusion) - so map only those ranges, not the full SIC
    taxonomy, to avoid inventing sector labels the engine never uses."""
    if sic is None:
        return None
    if 6500 <= sic <= 6553:
        return "Real Estate"
    if 6000 <= sic <= 6799:
        return "Financial Services"
    return None


def fetch_sec_profile(ticker: str) -> tuple[str | None, str | None]:
    """Sector/industry fallback via SEC EDGAR when Yahoo's quoteSummary crumb
    is blocked. Free, official, no auth — but US SEC registrants only, so
    foreign ordinary shares and non-US ETFs/funds (e.g. EQQQ.L) get nothing,
    same as they would from a blocked quoteSummary."""
    cik = _sec_cik_map().get(ticker.upper().split(".")[0])
    if cik is None:
        return None, None
    try:
        r = _yf_session().get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                              timeout=10, headers=_SEC_UA)
        d = r.json()
        sic = int(d["sic"]) if d.get("sic") else None
        return _sic_sector(sic), d.get("sicDescription")
    except Exception:  # noqa: BLE001 - best-effort; fields stay None
        return None, None


_SEC_REVENUE_TAGS = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                     "RevenueFromContractWithCustomerIncludingAssessedTax")
_SEC_NET_INCOME_TAGS = ("NetIncomeLoss",)
_SEC_SHARES_TAG = "EntityCommonStockSharesOutstanding"  # dei taxonomy, instant, cover-page fact


def fetch_sec_companyfacts(ticker: str) -> dict | None:
    """Raw companyfacts JSON for a US-listed ticker (official filings, no
    crumb/rate-limit risk), or None for a non-US ticker / unresolved CIK /
    unreachable endpoint - always best-effort, never fatal. Public: called
    once per report by fetch_sec_xbrl_fundamentals AND by anchors.py's
    own-history percentiles, so both share one fetch of the ~1-5MB payload
    rather than each re-requesting it per concept."""
    cik = _sec_cik_map().get(ticker.upper().split(".")[0])
    if cik is None:
        return None
    try:
        r = _yf_session().get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                              timeout=15, headers=_SEC_UA)
        return r.json()
    except Exception:  # noqa: BLE001 - best-effort; caller falls back to Yahoo
        return None


def _sec_by_duration(facts: dict, tags: tuple[str, ...]
                     ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """(quarterly, annual) dicts keyed by (start, end) date-string pairs,
    picked purely by each record's own duration (~91d / ~365d) - not by the
    SEC API's fy/fp fields. Live-checked on AAPL: fy/fp identify "as
    reported in fiscal-year X's 10-K/10-Q", but a single 10-K tags ALL THREE
    of its income-statement columns (current year + two comparatives) with
    the SAME (fy, fp='FY'), so bucketing by (fy, fp) silently picks an
    arbitrary one of the three real years. Keying by the record's actual
    (start, end) avoids that collision entirely and also de-dupes a period
    that shows up again as a comparative in a later filing."""
    quarterly: dict[tuple[str, str], float] = {}
    annual: dict[tuple[str, str], float] = {}
    for tag in tags:
        for rec in facts.get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", []):
            start, end, val = rec.get("start"), rec.get("end"), _f(rec.get("val"))
            if not start or not end or val is None:
                continue
            try:
                duration = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
            except ValueError:
                continue
            key = (start, end)
            if 80 <= duration <= 100:
                quarterly.setdefault(key, val)
            elif 350 <= duration <= 380:
                annual.setdefault(key, val)
    return quarterly, annual


def _sec_periods(facts: dict, tags: tuple[str, ...]) -> dict[str, float]:
    """{end_date: value} for every discrete quarterly period this concept
    resolves to, direct + Q4-derived, keyed by end date only (one value per
    real calendar quarter). Q1-Q3 are direct 10-Q discrete-quarter facts;
    Q4 = FY_total - Q1 - Q2 - Q3 for that fiscal year (a 10-K reports only
    the full year, never a discrete Q4) - found by chaining quarter
    start == previous end from the annual period's own start date, not by
    fy/fp (see _sec_by_duration). Shared building block for _sec_ttm (last
    4 / prior 4) and anchors.py's full rolling-TTM history."""
    quarterly, annual = _sec_by_duration(facts, tags)
    by_start = {start: (end, val) for (start, end), val in quarterly.items()}
    periods: dict[str, float] = {end: val for (_, end), val in quarterly.items()}
    for (fy_start, fy_end), fy_val in annual.items():
        if fy_end in periods:
            continue  # already have a discrete quarter at this end date
        chain, cur = [], fy_start
        for _ in range(3):
            nxt = by_start.get(cur)
            if nxt is None:
                break
            chain.append(nxt[1])
            # SEC's own quarter boundaries are contiguous but not
            # overlapping (live-checked on AAPL: Q1 ends 2025-12-27, Q2
            # starts 2025-12-28) - the next quarter's start is one day
            # after this one's end, never equal to it.
            cur = (datetime.fromisoformat(nxt[0]) + timedelta(days=1)).date().isoformat()
        if len(chain) == 3:
            periods[fy_end] = fy_val - sum(chain)
    return periods


def _sec_ttm(facts: dict, tags: tuple[str, ...]) -> tuple[float | None, float | None, str | None]:
    """Trailing-4-quarter sum plus the prior-year TTM (for a '_prev' YoY
    pairing). Needs >=4 resolved quarters on the trailing edge; returns
    (None, None, None) otherwise and the caller keeps Yahoo's TTM."""
    ordered = sorted(_sec_periods(facts, tags).items())  # ISO strings sort chronologically
    if len(ordered) < 4:
        return None, None, None
    ttm = sum(v for _, v in ordered[-4:])
    ttm_prev = sum(v for _, v in ordered[-8:-4]) if len(ordered) >= 8 else None
    return ttm, ttm_prev, f"SEC XBRL, TTM through quarter ended {ordered[-1][0]}"


def sec_ttm_history(facts: dict, tags: tuple[str, ...]) -> list[tuple[str, float]]:
    """Every trailing-4-quarter window over this concept's full XBRL history
    (chronological [(quarter_end, ttm_value), ...], not just the most
    recent) - anchors.py's own-history percentiles build P/S / EV/FCF
    distributions from this."""
    ordered = sorted(_sec_periods(facts, tags).items())
    return [(ordered[i][0], sum(v for _, v in ordered[i - 3:i + 1]))
           for i in range(3, len(ordered))]


def _sec_latest_shares(facts: dict) -> tuple[float | None, str | None]:
    """Most recent cover-page share count (dei taxonomy, instant fact) -
    typically fresher than Yahoo's cached sharesOutstanding since it's tied
    to the latest filing date rather than a periodic data-vendor refresh."""
    recs = facts.get("dei", {}).get(_SEC_SHARES_TAG, {}).get("units", {}).get("shares", [])
    valid = [(r["end"], v) for r in recs if r.get("end") and (v := _f(r.get("val"))) is not None]
    if not valid:
        return None, None
    end, val = max(valid, key=lambda t: t[0])
    return val, f"SEC XBRL dei:{_SEC_SHARES_TAG}, as of {end}"


def fetch_sec_xbrl_fundamentals(ticker: str) -> dict:
    """Official SEC XBRL companyfacts as a primary source for US filers:
    trailing-4-quarter revenue/net income (see _sec_ttm) and the latest
    cover-page share count. Every field isolated and best-effort; returns
    {} for a non-US ticker, an unresolved CIK, or a filer with too little
    quarterly XBRL history - callers treat this as optional, with Yahoo
    always the fallback."""
    doc = fetch_sec_companyfacts(ticker)
    if doc is None:
        return {}
    facts = doc.get("facts", {})
    out: dict = {}
    rev_ttm, rev_prev, rev_src = _sec_ttm(facts, _SEC_REVENUE_TAGS)
    if rev_ttm is not None:
        out["revenue"], out["revenue_prev"], out["revenue_source"] = rev_ttm, rev_prev, rev_src
    ni_ttm, ni_prev, ni_src = _sec_ttm(facts, _SEC_NET_INCOME_TAGS)
    if ni_ttm is not None:
        out["net_income"], out["net_income_prev"], out["net_income_source"] = ni_ttm, ni_prev, ni_src
    shares, shares_src = _sec_latest_shares(facts)
    if shares is not None:
        out["shares"], out["shares_source"] = shares, shares_src
    return out


def _warn(f: dict, msg: str) -> None:
    """Append to data_warning rather than overwrite it, so an SEC-vs-Yahoo
    disagreement and the marketCap-vs-price*shares disagreement can both
    surface even when they fire on the same report."""
    f["data_warning"] = f"{f['data_warning']}; {msg}" if f.get("data_warning") else msg


def _apply_sec_xbrl_primary(f: dict, ticker: str) -> None:
    """US filers only: SEC XBRL companyfacts becomes the primary source for
    revenue/net_income/shares (official, audited, no crumb/rate-limit risk)
    whenever it resolves a full trailing-4-quarter window; Yahoo's TTM (see
    _ttm_frame) stays the fallback for non-US tickers or thin XBRL history.
    Independently flags >10% disagreement between the two sources, same
    threshold/pattern as the marketCap-vs-price*shares check."""
    xbrl = fetch_sec_xbrl_fundamentals(ticker)
    switched = []
    for field in ("revenue", "net_income", "shares"):
        sec_val = xbrl.get(field)
        if sec_val is None:
            continue
        yahoo_val = f.get(field)
        if yahoo_val:
            disagree = abs(sec_val - yahoo_val) / abs(yahoo_val)
            if disagree > 0.10:
                _warn(f, f"SEC XBRL vs Yahoo {field} disagree by {disagree:.0%}")
        f[field], f[field + "_source"] = sec_val, xbrl[field + "_source"]
        if field + "_prev" in xbrl:
            f[field + "_prev"] = xbrl[field + "_prev"]
        switched.append(field)
    if switched:
        f["statement_basis"] += f" + SEC XBRL primary ({'/'.join(switched)})"


_BENCHMARK_BY_SUFFIX = (
    (".L", "^FTSE"), (".AS", "^AEX"), (".DE", "^GDAXI"), (".PA", "^FCHI"),
    (".TO", "^GSPTSE"), (".HK", "^HSI"), (".T", "^N225"),
)

# Wrapper ETFs/ETCs whose LISTING venue does not match their UNDERLYING
# exposure - suffix-based picking would give a Nasdaq-100 UCITS ETF (EQQQ.L,
# CNX1.L) a FTSE beta, or a gold ETC (IGLN.L, SGLN.L) an equity beta at all,
# neither of which is the risk factor the instrument actually carries.
# Checked before the suffix table, not instead of it, so any ticker outside
# this short list still gets the (rougher) suffix fallback rather than ^GSPC.
_BENCHMARK_OVERRIDE_BY_TICKER = {
    "EQQQ.L": "^NDX", "CNX1.L": "^NDX", "IGLN.L": "GC=F", "SGLN.L": "GC=F",
    "GLD": "GC=F", "IAU": "GC=F",
}


def benchmark_for(ticker: str) -> str:
    """Single source of truth for "what index/future is the right risk-factor
    comparison for this ticker", shared by compute_beta (cost-of-equity) and
    track_accuracy (benchmark-relative scoring) - named override first, then
    exchange suffix, defaulting to ^GSPC."""
    upper = ticker.upper()
    return _BENCHMARK_OVERRIDE_BY_TICKER.get(upper) or next(
        (idx for suf, idx in _BENCHMARK_BY_SUFFIX if upper.endswith(suf)), "^GSPC")


def macro_asset_class(ticker: str) -> str:
    """Which macro tilt applies (macro_rules.regime_and_tilt): "gold" for
    gold itself and gold-tracking products, "tech" (general equity risk
    appetite) for everything else. GC=F is named directly because its
    benchmark cannot be itself."""
    return "gold" if ticker.upper() == "GC=F" or benchmark_for(ticker) == "GC=F" else "tech"


def compute_beta(ticker: str, period: str = "3y") -> tuple[float | None, str | None]:
    """beta = Cov(r_stock, r_bench) / Var(r_bench) on weekly returns, computed
    directly from price history (chart endpoint, crumb-free) instead of
    reading Yahoo's own published beta (quoteSummary, crumb-gated). Benchmark
    picked by a named ticker override first, then exchange suffix; not
    currency-hedged, so cross-listed tickers get a rough estimate rather than
    Yahoo's exact methodology - a named tradeoff, but still more grounded
    than defaulting every unknown-beta firm to 1.0 in cost-of-equity."""
    bench = benchmark_for(ticker)
    try:
        s_hist = _ticker(ticker).history(period=period, interval="1wk", auto_adjust=True)["Close"]
        b_hist = _ticker(bench).history(period=period, interval="1wk", auto_adjust=True)["Close"]
        joined = pd.concat([s_hist, b_hist], axis=1, keys=["s", "b"]).dropna()
        if len(joined) < 26:  # need a meaningful sample, not a handful of weeks
            return None, None
        rs, rb = joined["s"].pct_change().dropna(), joined["b"].pct_change().dropna()
        rs, rb = rs.align(rb, join="inner")
        var_b = rb.var()
        if not var_b:
            return None, None
        return float(rs.cov(rb) / var_b), f"computed vs {bench} weekly returns, n={len(rs)} (quoteSummary beta blocked)"
    except Exception:  # noqa: BLE001 - best-effort; caller keeps beta=None
        return None, None


def _f(v) -> float | None:
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _row(df: pd.DataFrame, names: tuple[str, ...], col: int) -> float | None:
    if df is None or df.empty or col >= df.shape[1]:
        return None
    for n in names:
        if n in df.index:
            return _f(df.loc[n].iloc[col])
    return None


def _http_get(url: str, timeout: float = 10.0) -> str:
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:  # noqa: S310
        return r.read().decode(errors="replace")


def fetch_risk_free_rate() -> tuple[float, str]:
    """Most recent 10Y par yield. Treasury.gov's official daily par yield
    curve CSV (no key) -> yfinance ^TNX -> hardcoded fallback. Each source is
    tried in isolation and never raises; the caller always gets a usable
    (rate, source) pair so a report can say exactly which one was used."""
    year = datetime.now(timezone.utc).year
    url = ("https://home.treasury.gov/resource-center/data-chart-center/"
           f"interest-rates/daily-treasury-rates.csv/{year}/all"
           f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")
    try:
        rows = list(csv.reader(io.StringIO(_http_get(url))))
        header, latest = rows[0], rows[1]
        rate = _f(latest[header.index("10 Yr")])
        if rate is not None:
            return rate / 100.0, f"treasury.gov 10Y par yield ({latest[0]})"
    except Exception:  # noqa: BLE001 - fall through to next source
        pass
    try:
        hist = _ticker("^TNX").history(period="5d")
        if hist is not None and not hist.empty:
            rate = _f(hist["Close"].iloc[-1])
            if rate is not None:
                return rate / 100.0, "yfinance ^TNX (CBOE 10Y yield index, fallback)"
    except Exception:  # noqa: BLE001
        pass
    return RF_FALLBACK, f"fallback constant {RF_FALLBACK:.1%} (live sources unreachable)"


def fetch_implied_erp() -> tuple[float, str]:
    """Damodaran's monthly S&P 500 implied ERP (trailing-12m, adjusted-payout
    method), scraped from his NYU Stern homepage since he does not publish a
    stable JSON/CSV endpoint. Numbers there are split across HTML tags (e.g.
    "4.<span>...</span>28%"), hence the tag-strip before the regex. Falls
    back to a constant if the page layout changes or is unreachable."""
    try:
        text = re.sub(r"<[^>]+>", "", _http_get(
            "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm")).replace("&amp;", "&")
        m = re.search(r"Implied ERP on ([A-Za-z]+ \d{1,2}, \d{4})\s*=\s*([\d.]+)%", text)
        if m:
            return float(m.group(2)) / 100.0, f"Damodaran implied ERP, {m.group(1)}"
    except Exception:  # noqa: BLE001
        pass
    return ERP_FALLBACK, f"fallback constant {ERP_FALLBACK:.1%} (Damodaran page unavailable)"


def fetch_macro() -> dict:
    rf, rf_source = fetch_risk_free_rate()
    erp, erp_source = fetch_implied_erp()
    return {"rf": rf, "rf_source": rf_source, "erp": erp, "erp_source": erp_source}


_PENCE_CURRENCIES = ("GBp", "GBX")


def _price_scale(currency: str | None) -> tuple[float, str | None]:
    """Yahoo quotes some LSE lines in pence (GBp/GBX) while price/marketCap/
    EPS/technicals all ride the same raw feed - live-checked via fast_info:
    EQQQ.L and BARC.L both report currency 'GBp' (BARC.L lastPrice 483.45 =
    GBP 4.8345), while IGLN.L reports 'USD' despite the same .L suffix. So
    this must be per-instrument (fast_info), never per-suffix. Mixing an
    un-rescaled pence price into a per-share model silently misprices
    everything downstream by ~100x. Every other currency passes through
    unchanged."""
    if currency in _PENCE_CURRENCIES:
        return 0.01, "GBP"
    return 1.0, currency


def _ttm_frame(quarterly: pd.DataFrame, annual: pd.DataFrame, is_flow: bool
              ) -> tuple[pd.DataFrame, "pd.Timestamp | None"]:
    """2-column frame consumed by fetch_fundamentals' existing col/col+1
    extraction loops: column 0 = TTM (last 4 quarters summed for a flow
    statement - income/cashflow - or the single latest quarter for a
    balance sheet, which is a point-in-time snapshot, never a sum); column 1
    = the most recently completed full fiscal year from the annual
    statement, the standard "TTM vs last fiscal year" pairing Piotroski/
    Mohanram/CCC's prior-period fields need (real analysts use this same
    pairing - it works with only 4 quarters of quarterly history instead of
    requiring 8). Built with pd.concat (positional), not a dict, so a
    quarter-end that happens to land on the same date as the fiscal
    year-end can never collide and silently drop one side."""
    if quarterly is None or quarterly.empty or quarterly.shape[1] < 4:
        return pd.DataFrame(), None
    ttm = (quarterly.iloc[:, 0:4].sum(axis=1, min_count=1) if is_flow
          else quarterly.iloc[:, 0]).rename(0)
    parts = [ttm]
    if annual is not None and not annual.empty:
        parts.append(annual.iloc[:, 0].rename(1))
    return pd.concat(parts, axis=1), quarterly.columns[0]


def fetch_fundamentals(ticker: str, col: int = 0) -> dict:
    """Map yfinance statements into the valuation field dict. Missing fields
    stay None -> models exclude themselves with named reasons downstream.

    col selects which annual statement column is "current" (0 = latest, 1 =
    one year back, ...); col+1 becomes "prev". backtest.py reuses this with
    col > 0 to reconstruct point-in-time fundamentals for historical walk-
    forward testing instead of duplicating the field-mapping table there."""
    t = _ticker(ticker)
    info = t.info or {}
    info_price = _f(info.get("currentPrice") or info.get("regularMarketPrice"))
    fast: dict = {}
    if col == 0 and info_price is None:
        # .info is backed by quoteSummary, which needs a crumb; some networks
        # block/rate-limit that handshake even with browser impersonation.
        # fast_info hits a separate, crumb-free endpoint and recovers the
        # handful of fields (price/market_cap/shares) every model needs to
        # run at all, so a blocked crumb degrades quality instead of killing
        # the whole valuation.
        try:
            fi = t.fast_info
            fast = {"price": _f(fi.get("lastPrice")), "market_cap": _f(fi.get("marketCap")),
                    "shares": _f(fi.get("shares")), "quote_type": fi.get("quoteType"),
                    "currency": fi.get("currency")}
        except Exception:  # noqa: BLE001 - best-effort; fields stay None
            pass
    annual_inc, annual_bal, annual_cf = t.income_stmt, t.balance_sheet, t.cashflow
    inc, bal, cf = annual_inc, annual_bal, annual_cf
    # Annual statements can be a year stale (valuing AAPL in Aug on FY-Sep
    # annuals) - prefer TTM built from quarterly data for the CURRENT period
    # only (col==0); backtest.py's historical reconstruction (col>0) keeps
    # reading annual columns, since quarterly history isn't deep enough for
    # multi-year walk-forward anyway.
    statement_basis, ttm_date = "annual (fiscal year)", None
    if col == 0:
        try:
            ttm_inc, ttm_date = _ttm_frame(t.quarterly_income_stmt, annual_inc, is_flow=True)
            ttm_cf, _ = _ttm_frame(t.quarterly_cashflow, annual_cf, is_flow=True)
            ttm_bal, _ = _ttm_frame(t.quarterly_balance_sheet, annual_bal, is_flow=False)
            # Flows (revenue/EBIT/NI/CFO) drive every model - require both to
            # resolve before switching basis at all. Balance sheet is allowed
            # to fall back to annual on its own (labeled), since a thin
            # quarterly balance sheet shouldn't block the fresher income read.
            if not ttm_inc.empty and not ttm_cf.empty:
                inc, cf = ttm_inc, ttm_cf
                if not ttm_bal.empty:
                    bal, statement_basis = ttm_bal, "TTM (trailing 4 quarters)"
                else:
                    statement_basis = "TTM income/cashflow, latest annual balance sheet"
        except Exception:  # noqa: BLE001 - best-effort; annual stays the fallback
            pass
    scale, currency = _price_scale(info.get("currency") or fast.get("currency"))

    def _scaled(v: float | None) -> float | None:
        return v * scale if v is not None else None

    price_raw = info_price or fast.get("price")
    mcap_raw = _f(info.get("marketCap")) or fast.get("market_cap")
    eps_fwd_raw = _f(info.get("forwardEps"))
    price_src = None
    if price_raw is not None:
        price_src = "yfinance quoteSummary" if info_price else "fast_info (quoteSummary crumb blocked on this network)"
        if scale != 1.0:
            price_src += " (converted from GBX/GBp pence)"

    f: dict = {
        "quote_type": info.get("quoteType") or fast.get("quote_type"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": currency if col == 0 else None,
        "price": _scaled(price_raw) if col == 0 else None,
        "market_cap": _scaled(mcap_raw) if col == 0 else None,
        "beta": _f(info.get("beta")),
        "shares": (_f(info.get("sharesOutstanding")) or fast.get("shares")) if col == 0 else None,
        "eps_fwd": _scaled(eps_fwd_raw) if col == 0 else None,
        "name": info.get("shortName") or ticker,
        "statement_date": None,
        "statement_basis": statement_basis,
        "price_source": price_src if col == 0 else None,
    }
    if inc is not None and not inc.empty and col < inc.shape[1]:
        stmt_date = ttm_date if ttm_date is not None else inc.columns[col]
        f["statement_date"] = str(stmt_date.date())
    if col == 0 and f["statement_date"]:
        f["statement_days_stale"] = (datetime.now(timezone.utc).date()
                                     - datetime.strptime(f["statement_date"], "%Y-%m-%d").date()).days

    for field, names in {
        "revenue": ("Total Revenue",), "ebit": ("EBIT", "Operating Income"),
        "net_income": ("Net Income",), "pretax_income": ("Pretax Income",),
        "tax_provision": ("Tax Provision",), "gross_profit": ("Gross Profit",),
        "rnd": ("Research And Development",),
        "sga": ("Selling General And Administration",),
        "interest_expense": ("Interest Expense", "Interest Expense Non Operating"),
        "ebit_unusual_items": ("Total Unusual Items",),
    }.items():
        f[field] = _row(inc, names, col)
        f[field + "_prev"] = _row(inc, names, col + 1)

    for field, names in {
        "total_assets": ("Total Assets",),
        "total_liabilities": ("Total Liabilities Net Minority Interest",),
        "current_assets": ("Current Assets",),
        "current_liabilities": ("Current Liabilities",),
        "cash": ("Cash And Cash Equivalents",),
        "total_debt": ("Total Debt",), "ltd": ("Long Term Debt",),
        "equity": ("Stockholders Equity",),
        "retained_earnings": ("Retained Earnings",),
        "receivables": ("Receivables", "Accounts Receivable"),
        "inventory": ("Inventory",),
        "payables": ("Accounts Payable", "Payables And Accrued Expenses"),
        "net_ppe": ("Net PPE",),
        "shares_bs": ("Ordinary Shares Number",),
    }.items():
        f[field] = _row(bal, names, col)
        f[field + "_prev"] = _row(bal, names, col + 1)

    f["cfo"] = _row(cf, ("Operating Cash Flow",), col)
    f["cfo_prev"] = _row(cf, ("Operating Cash Flow",), col + 1)
    f["capex"] = _row(cf, ("Capital Expenditure",), col)
    f["depreciation"] = _row(cf, ("Depreciation And Amortization",), col)
    f["depreciation_prev"] = _row(cf, ("Depreciation And Amortization",), col + 1)
    f["fcf"] = _row(cf, ("Free Cash Flow",), col)
    buyback_cash = _row(cf, ("Repurchase Of Capital Stock",), col)
    f["buybacks"] = abs(buyback_cash) if buyback_cash is not None else None
    f["analyst_target_mean"] = _scaled(_f(info.get("targetMeanPrice"))) if col == 0 else None
    f["analyst_target_median"] = _scaled(_f(info.get("targetMedianPrice"))) if col == 0 else None
    f["analyst_target_count"] = _f(info.get("numberOfAnalystOpinions")) if col == 0 else None

    # prefer balance-sheet share count for prev-year dilution check
    if f.get("shares_bs"):
        f["shares"] = f["shares_bs"] or f["shares"]
        f["shares_prev"] = f.get("shares_bs_prev")
    if not f.get("net_income_prev"):
        f["net_income_prev"] = None

    # 5y revenue CAGR always reads the real multi-year annual columns, even
    # when the current-period fields above came from a 2-column TTM frame
    # (annual_inc is never swapped) - a CAGR needs actual years of history,
    # not a TTM-vs-LFY pair.
    if annual_inc is not None and not annual_inc.empty and "Total Revenue" in annual_inc.index:
        rev = [x for x in (_f(v) for v in annual_inc.loc["Total Revenue"].iloc[col:]) if x]
        if len(rev) >= 2 and rev[-1] > 0:
            yrs = len(rev) - 1
            f["revenue_growth_5y"] = (rev[0] / rev[-1]) ** (1 / yrs) - 1
    if col > 0:
        # Backtest snapshot: current-quote-only fields (growth_est, analyst
        # targets, live price/shares) don't exist as-of a past fiscal year in
        # yfinance's info blob, so leave them unset rather than leaking today's
        # value into a historical fundamentals reconstruction.
        return f
    # AEG's growth input: prefer analysts' 5y consensus EPS CAGR (matches the
    # textbook AEG data source - "analysts' consensus earnings forecasts") over
    # yfinance's earningsGrowth, which is a single next-FY estimate, not a CAGR.
    growth_est, growth_source = None, None
    try:
        ge = t.growth_estimates
        if ge is not None and not ge.empty:
            row = next((r for r in ("+5y", "+5Y") if r in ge.index), None)
            gcol = next((c for c in ("stockTrend", "stock") if c in ge.columns), None)
            if row and gcol:
                growth_est = _f(ge.loc[row, gcol])
                if growth_est is not None:
                    growth_source = "analyst consensus 5y EPS CAGR (yfinance growth_estimates)"
    except Exception:  # noqa: BLE001 - best-effort; fall through to next source
        pass
    if growth_est is None:
        growth_est = _f(info.get("earningsGrowth"))
        if growth_est is not None:
            growth_source = "yfinance earningsGrowth (single next-FY estimate, not a CAGR)"
    f["growth_est"] = growth_est
    f["growth_est_source"] = growth_source

    if f["beta"] is None:
        # quoteSummary beta blocked -> compute it ourselves from the same
        # crumb-free chart endpoint price history already fetched elsewhere.
        f["beta"], f["beta_source"] = compute_beta(ticker)
    else:
        f["beta_source"] = "yfinance quoteSummary"

    if f["sector"] is None and f["industry"] is None:
        sec_sector, sec_industry = fetch_sec_profile(ticker)
        if sec_sector or sec_industry:
            f["sector"], f["industry"] = sec_sector, sec_industry
            f["sector_source"] = "SEC EDGAR SIC classification (quoteSummary sector/industry blocked)"
    else:
        f["sector_source"] = "yfinance quoteSummary" if f["sector"] else None

    if col == 0:
        _apply_sec_xbrl_primary(f, ticker)

    # two-source sanity: yfinance info marketCap vs price*shares (post-SEC,
    # so this checks against the freshest share count actually being used)
    if f["price"] and f["shares"]:
        implied = f["price"] * f["shares"]
        if f["market_cap"] and abs(implied - f["market_cap"]) / f["market_cap"] > 0.10:
            _warn(f, f"marketCap and price*shares disagree by "
                     f"{abs(implied - f['market_cap']) / f['market_cap']:.0%}")
    return f


def spot_price(ticker: str) -> float | None:
    """Cheap current price only (fast_info, crumb-free), scaled like every
    other price field here (see _price_scale) - for callers that just need
    "what is this worth right now" without a full fundamentals/technicals
    fetch (portfolio.py's holdings valuation)."""
    try:
        fi = _ticker(ticker).fast_info
        price = _f(fi.get("lastPrice"))
        if price is None:
            return None
        scale, _ = _price_scale(fi.get("currency"))
        return price * scale
    except Exception:  # noqa: BLE001 - best-effort; caller treats None as "price unavailable"
        return None


def fetch_technicals(ticker: str) -> dict:
    """Chart-endpoint Close prices ride the same raw pence feed as fast_info
    for GBX/GBp names (verified: EQQQ.L's chart Close and fast_info.lastPrice
    are the same ~53000 scale), so every price-denominated output here needs
    the same conversion fetch_fundamentals applies, or sma50/high_52w/MACD
    silently sit 100x off from the (correctly scaled) price the report shows
    next to them."""
    try:
        scale, _ = _price_scale(_ticker(ticker).fast_info.get("currency"))
    except Exception:  # noqa: BLE001 - best-effort; treat as already-correct scale
        scale = 1.0
    hist = _ticker(ticker).history(period="1y", auto_adjust=True)
    if hist is None or hist.empty:
        return {"error": "no price history"}
    c = hist["Close"] * scale
    out = {
        "price": _f(c.iloc[-1]),
        "chg_1m_pct": _f((c.iloc[-1] / c.iloc[-21] - 1) * 100) if len(c) > 21 else None,
        "chg_3m_pct": _f((c.iloc[-1] / c.iloc[-63] - 1) * 100) if len(c) > 63 else None,
        "chg_1y_pct": _f((c.iloc[-1] / c.iloc[0] - 1) * 100),
        "sma50": _f(c.rolling(50).mean().iloc[-1]),
        "sma200": _f(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None,
        "vol_ann_pct": _f(c.pct_change().std() * math.sqrt(252) * 100),
        "high_52w": _f(c.max()), "low_52w": _f(c.min()),
    }
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] else None
    out["rsi14"] = _f(100 - 100 / (1 + rs)) if rs else None
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    out["macd"] = _f(macd.iloc[-1])
    out["macd_signal"] = _f(macd.ewm(span=9).mean().iloc[-1])
    return out


def fetch_earnings_calendar(ticker: str) -> dict:
    """Next earnings date + recent realized EPS surprise history, straight
    from Yahoo's public earnings-calendar page (finance.yahoo.com/calendar/
    earnings?symbol=X) - a plain HTML scrape (yfinance's get_earnings_dates),
    NOT the quoteSummary crumb endpoint this network blocks elsewhere in this
    file, so this stays free/structured/no-LLM even on a run where growth_est/
    eps_fwd/analyst_target need the web-search fallback. No true 90-day
    estimate-revision *history* exists here or in any free source (that is a
    paid IBES/Zacks-style product) - agents_llm.fetch_market_consensus's
    existing web-search call best-effort asks for a qualitative revision
    read instead, tagged unverified like every other web-search field."""
    try:
        df = _ticker(ticker).get_earnings_dates(limit=6)
    except Exception as e:  # noqa: BLE001 - optional enrichment, never pipeline-ending
        return {"error": f"{e.__class__.__name__}: {str(e)[:120]}"}
    if df is None or df.empty:
        return {}
    out: dict = {}
    upcoming = df[df["Reported EPS"].isna()]
    if not upcoming.empty:
        next_date = upcoming.index[0].date()
        out["next_earnings_date"] = next_date.isoformat()
        out["next_earnings_eps_estimate"] = _f(upcoming.iloc[0]["EPS Estimate"])
        days_away = (next_date - datetime.now(timezone.utc).date()).days
        out["next_earnings_days_away"] = days_away
        # trader/risk seats' cue to size down or widen invalidation around an
        # imminent print - a 2-week window, not a precise vol-event model.
        out["flag_earnings_within_2w"] = 0 <= days_away <= 14
    reported = df[df["Reported EPS"].notna()].head(4)
    if not reported.empty:
        out["recent_earnings_surprises"] = [
            {"date": idx.date().isoformat(), "eps_estimate": _f(row["EPS Estimate"]),
             "eps_actual": _f(row["Reported EPS"]), "surprise_pct": _f(row["Surprise(%)"])}
            for idx, row in reported.iterrows()
        ]
        last2 = [_f(row["Surprise(%)"]) for _, row in reported.head(2).iterrows()]
        if all(s is not None for s in last2):
            out["flag_recent_earnings_misses"] = all(s < 0 for s in last2)
    return out


def _parse_news_date(when) -> datetime | None:
    if isinstance(when, (int, float)):
        try:
            return datetime.fromtimestamp(when, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(when, str) and when:
        try:
            return datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _headline_is_relevant(title: str, ticker: str, company_name: str | None) -> bool:
    """Live-checked on AAPL: yfinance's per-ticker news list carries no
    relatedTickers metadata on this feed, yet mixes in genuinely off-topic
    stories (a "how many employees does UnitedHealth have" piece showed up
    under AAPL). A keyword check against the ticker symbol / company name is
    the best available signal - kept deliberately permissive (only drops on
    an explicit non-match against a known company name) since a false
    negative here just loses one headline, while a false positive would
    silently discard real signal."""
    t = title.lower()
    symbol = ticker.split(".")[0].lower()
    if symbol and symbol in t:
        return True
    if not company_name:
        return True  # nothing to check against -> don't filter
    first_word = re.split(r"[^A-Za-z0-9]+", company_name)[0].lower()
    return len(first_word) >= 3 and first_word in t


def fetch_news(ticker: str, limit: int = 8, company_name: str | None = None,
              max_age_days: int = 30) -> list[dict]:
    """Headlines only — untrusted text, passed downstream as data. Filtered
    for recency (a month-old headline mixed into a "current" pack reads as
    fresh unless the LLM happens to notice the printed date) and relevance
    (see _headline_is_relevant)."""
    items: list[dict] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    try:
        for n in (_ticker(ticker).news or []):
            content = n.get("content") or n
            title = content.get("title")
            if not title:
                continue
            when = _parse_news_date(content.get("pubDate") or content.get("providerPublishTime"))
            if when is not None and when < cutoff:
                continue
            if not _headline_is_relevant(str(title), ticker, company_name):
                continue
            items.append({"title": str(title)[:200],
                         "date": when.strftime("%Y-%m-%d") if when else ""})
            if len(items) >= limit:
                break
    except Exception as e:  # noqa: BLE001 - news is best-effort
        items.append({"title": f"(news fetch failed: {e.__class__.__name__})", "date": ""})
    return items


def fetch_etf_extras(ticker: str) -> dict:
    """ETF regime: holdings concentration + options skew via OpenBB when
    available; every miss is named, never fatal."""
    out: dict = {}
    t = _ticker(ticker)
    try:
        fd = t.funds_data
        top = fd.top_holdings
        if top is not None and not top.empty:
            w = top.iloc[:, -1]
            out["top10_weight_pct"] = _f(w.head(10).sum() * 100)
            out["top_holdings"] = [f"{i} {round(float(w.loc[i]) * 100, 1)}%"
                                   for i in top.index[:5]]
    except Exception as e:  # noqa: BLE001
        out["holdings_note"] = f"holdings unavailable ({e.__class__.__name__})"
    try:
        exps = t.options
        if exps:
            chain = t.option_chain(exps[0])
            put_oi = chain.puts["openInterest"].sum()
            call_oi = chain.calls["openInterest"].sum()
            if call_oi:
                out["put_call_oi_ratio"] = _f(put_oi / call_oi)
            out["options_expiry_used"] = exps[0]
    except Exception as e:  # noqa: BLE001
        out["options_note"] = f"options unavailable ({e.__class__.__name__})"
    return out


def gather(ticker: str) -> dict:
    f = fetch_fundamentals(ticker)
    pack = {
        "ticker": ticker.upper(),
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "fundamentals": f,
        "macro": fetch_macro(),
        "technicals": fetch_technicals(ticker),
        "news": fetch_news(ticker, company_name=f.get("name")),
    }
    if (f.get("quote_type") or "").upper() in ("ETF", "MUTUALFUND"):
        pack["etf"] = fetch_etf_extras(ticker)
    return pack

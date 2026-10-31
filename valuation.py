"""Deterministic valuation + quality engine. No network, no LLM.

All functions take plain dicts of fundamentals (see FIELDS below) so they are
golden-testable. Every model returns {"value": per-share float | None,
"applicable": bool, "reason": str, "details": {...}} — a model that cannot run
EXCLUDES itself with a named reason; it never silently returns 0 (silent zeros
skew ensemble weights, the failure mode we rejected in ai-hedge-fund's design).

Model choice basis: Penman & Sougiannis (1998) / Francis, Olsson & Oswald (2000):
residual-income methods have the lowest pricing errors over finite horizons
because value anchors on the balance sheet, not an ad-hoc terminal guess.
"""

from __future__ import annotations

import math
import random

# Fundamentals dict fields (all floats or None; latest fiscal year unless _prev):
# revenue, revenue_prev, ebit, net_income, net_income_prev, pretax_income,
# tax_provision, gross_profit, gross_profit_prev, cfo, cfo_prev, capex,
# depreciation, total_assets, total_assets_prev, total_liabilities,
# current_assets, current_assets_prev, current_liabilities,
# current_liabilities_prev, cash, total_debt, total_debt_prev, ltd, ltd_prev,
# equity, retained_earnings, receivables, receivables_prev, shares,
# shares_prev, market_cap, price, beta, dividends_paid, fcf,
# eps_fwd, growth_est (analyst LT growth, decimal), rnd, sga, sga_prev,
# quote_type ("EQUITY"/"ETF"...), sector, revenue_growth_5y (CAGR, decimal)

RF_DEFAULT = 0.042  # 10y treasury fallback
ERP = 0.05
TERMINAL_G = 0.025
YEARS = 10


def _ok(*vals) -> bool:
    return all(isinstance(v, (int, float)) and not math.isnan(v) for v in vals)


def na(reason: str) -> dict:
    return {"value": None, "applicable": False, "reason": reason, "details": {}}


# ---------------------------------------------------------------- discount rates

def cost_of_equity(f: dict, rf: float = RF_DEFAULT, erp: float = ERP) -> float:
    beta = f.get("beta") if _ok(f.get("beta")) else 1.0
    return rf + beta * erp


def wacc(f: dict, rf: float = RF_DEFAULT, erp: float = ERP) -> float:
    ke = cost_of_equity(f, rf, erp)
    debt = f.get("total_debt") or 0.0
    mcap = f.get("market_cap") or 0.0
    if debt <= 0 or mcap <= 0:
        return ke
    kd = 0.05
    tax = effective_tax(f)
    wd = debt / (debt + mcap)
    return (1 - wd) * ke + wd * kd * (1 - tax)


def effective_tax(f: dict) -> float:
    if _ok(f.get("tax_provision"), f.get("pretax_income")) and f["pretax_income"] > 0:
        return min(max(f["tax_provision"] / f["pretax_income"], 0.0), 0.4)
    return 0.21


# ---------------------------------------------------------------- regime

STATEMENT_EVIDENCE_FIELDS = ("ebit", "revenue", "net_income", "total_assets", "total_liabilities")


def classify_regime(f: dict) -> dict:
    qt = (f.get("quote_type") or "").upper()
    if qt in ("ETF", "MUTUALFUND", "INDEX"):
        return {"regime": "etf", "reason": "fund/ETF: no firm fundamentals; use technicals/flows/macro"}
    if not any(_ok(f.get(k)) for k in STATEMENT_EVIDENCE_FIELDS):
        # No statement evidence at all, regardless of what quote_type
        # claims - live-checked: IGLN.L (a gold ETC) reports quote_type
        # "EQUITY" from fast_info despite having zero statements of any
        # kind, so quote_type is not a reliable signal here. Defaulting to
        # EQUITY/mature-profitable on zero evidence was the original
        # IGLN.L bug (tagged a going concern with 0% data coverage);
        # unknown is honest and tells the committee this is a no-view.
        return {"regime": "unknown", "reason": "no statement data available; cannot classify"}
    sector = (f.get("sector") or "").lower()
    if "real estate" in sector:
        return {"regime": "unsupported", "reason": "REIT: NAV-based valuation not implemented"}
    financial = "financial" in sector
    nopat = nopat_of(f)
    z = altman_z(f).get("details", {}).get("z")
    rev_g = f.get("revenue_growth_5y")
    if _ok(nopat) and nopat <= 0:
        if _ok(rev_g) and rev_g < 0 and _ok(z) and z < 1.81:
            return {"regime": "distressed", "reason": "negative NOPAT, shrinking revenue, Z < 1.81"}
        return {"regime": "growth-unprofitable", "reason": "negative NOPAT/earnings"}
    if financial:
        return {"regime": "financial", "reason": "financial sector: equity-RIM (NOA/WACC ill-defined)"}
    return {"regime": "mature-profitable", "reason": "positive NOPAT"}


def nopat_of(f: dict) -> float | None:
    if not _ok(f.get("ebit")):
        return None
    return f["ebit"] * (1 - effective_tax(f))


def noa_of(f: dict, prev: bool = False) -> float | None:
    """NOA = operating assets - operating liabilities
         = (total assets - cash) - (total liabilities - total debt)."""
    s = "_prev" if prev else ""
    ta, tl = f.get(f"total_assets{s}"), f.get("total_liabilities" if not prev else "total_liabilities_prev")
    cash = f.get("cash") if not prev else f.get("cash_prev", f.get("cash"))
    debt = f.get(f"total_debt{s}")
    if not _ok(ta, tl):
        return None
    cash = cash if _ok(cash) else 0.0
    debt = debt if _ok(debt) else 0.0
    return (ta - cash) - (tl - debt)


# ---------------------------------------------------------------- models

def reoi_value(f: dict, rf: float = RF_DEFAULT, erp: float = ERP,
               w: float | None = None, fade_floor: float = 0.25,
               terminal_g: float = TERMINAL_G) -> dict:
    """Enterprise residual operating income:
    EV = NOA0 + sum PV(NOPAT_t - WACC*NOA_{t-1}).
    RNOA spread fades linearly to fade_floor of initial by year YEARS
    (competitive decay), terminal = perpetuity of final ReOI at terminal_g.
    Equity = EV - net debt. fade_floor/terminal_g are parameterized (not just
    module constants) so monte_carlo_range() can perturb them per sample."""
    nopat, noa = nopat_of(f), noa_of(f)
    if not _ok(nopat, noa) or noa <= 0:
        return na("ReOI needs positive NOA and EBIT data")
    if nopat <= 0:
        return na("ReOI not meaningful with negative NOPAT")
    if not _ok(f.get("shares")) or f["shares"] <= 0:
        return na("missing share count")
    r = w if w is not None else wacc(f, rf, erp)
    rnoa0 = nopat / noa
    spread0 = rnoa0 - r
    g = min(f.get("revenue_growth_5y") or 0.04, 0.12)
    g = max(g, 0.0)
    pv, noa_t = 0.0, noa
    reoi_t = 0.0
    for t in range(1, YEARS + 1):
        fade = 1 - (1 - fade_floor) * (t - 1) / (YEARS - 1)  # 1.0 -> fade_floor
        spread_t = spread0 * fade
        reoi_t = spread_t * noa_t
        pv += reoi_t / (1 + r) ** t
        noa_t *= 1 + g * (1 - (t - 1) / YEARS)  # growth fades too
    term = 0.0
    if r > terminal_g and reoi_t > 0:
        term = reoi_t * (1 + terminal_g) / (r - terminal_g) / (1 + r) ** YEARS
    ev = noa + pv + term
    net_debt = (f.get("total_debt") or 0.0) - (f.get("cash") or 0.0)
    per_share = (ev - net_debt) / f["shares"]
    return {"value": per_share, "applicable": True, "reason": "ok",
            "details": {"rnoa": rnoa0, "wacc": r, "spread": spread0, "noa": noa,
                        "ev": ev, "terminal_pct": term / ev if ev else None}}


def equity_rim_value(f: dict, rf: float = RF_DEFAULT, erp: float = ERP,
                     fade_floor: float = 0.25, terminal_g: float = TERMINAL_G) -> dict:
    """Equity residual income for financials: BV + sum PV(NI_t - ke*BV_{t-1}).
    ROE spread fades like ReOI."""
    ni, bv = f.get("net_income"), f.get("equity")
    if not _ok(ni, bv) or bv <= 0:
        return na("RIM needs net income and positive book value")
    if ni <= 0:
        return na("RIM not meaningful with negative earnings")
    if not _ok(f.get("shares")) or f["shares"] <= 0:
        return na("missing share count")
    ke = cost_of_equity(f, rf, erp)
    roe0 = ni / bv
    spread0 = roe0 - ke
    payout = 0.4
    pv, bv_t, ri_t = 0.0, bv, 0.0
    for t in range(1, YEARS + 1):
        fade = 1 - (1 - fade_floor) * (t - 1) / (YEARS - 1)
        roe_t = ke + spread0 * fade
        ni_t = roe_t * bv_t
        ri_t = ni_t - ke * bv_t
        pv += ri_t / (1 + ke) ** t
        bv_t += ni_t * (1 - payout)
    term = 0.0
    if ke > terminal_g and ri_t > 0:
        term = ri_t * (1 + terminal_g) / (ke - terminal_g) / (1 + ke) ** YEARS
    val = bv + pv + term
    return {"value": val / f["shares"], "applicable": True, "reason": "ok",
            "details": {"roe": roe0, "ke": ke, "book_value": bv}}


def aeg_value(f: dict, rf: float = RF_DEFAULT, erp: float = ERP) -> dict:
    """Ohlson-Juettner abnormal earnings growth, anchored on forward EPS:
    V0 = (1/r) * [EPS1 + sum PV(AEG_t)],  AEG_t = EPS_t - EPS_{t-1}*(1+r) + r*DPS_{t-1}."""
    eps1 = f.get("eps_fwd")
    if not _ok(eps1) or eps1 <= 0:
        return na("AEG needs positive forward EPS")
    r = cost_of_equity(f, rf, erp)
    g = f.get("growth_est")
    g = min(g, 0.20) if _ok(g) else 0.06
    payout = 0.3
    pv_aeg, eps_prev = 0.0, eps1
    for t in range(2, 7):
        yrs_in = t - 2
        g_t = g + (TERMINAL_G - g) * yrs_in / 4  # fade to terminal by year 6
        eps_t = eps_prev * (1 + g_t)
        dps_prev = eps_prev * payout
        aeg_t = eps_t - eps_prev * (1 + r) + r * dps_prev
        pv_aeg += aeg_t / (1 + r) ** (t - 1)
        eps_prev = eps_t
    val = (eps1 + pv_aeg) / r
    if val <= 0:
        return na("AEG collapsed non-positive (abnormal growth deeply negative)")
    return {"value": val, "applicable": True, "reason": "ok",
            "details": {"eps_fwd": eps1, "ke": r, "growth_used": g}}


def owner_earnings_value(f: dict, rf: float = RF_DEFAULT, erp: float = ERP) -> dict:
    """Buffett owner earnings (ai-hedge-fund parametrization, cross-checked):
    OE = NI + D&A - capex - dWC; 5y projection + Gordon terminal; 25% MOS haircut.
    Conservative floor estimate for mature cash generators.

    Discount rate is the same live cost-of-equity every other model in this
    module uses (rf + beta*erp), not a fixed 15% hurdle - a hardcoded req
    ignored the rf/erp arguments entirely, so a name with a low beta/rf and
    one with a high beta/rf got priced off the same made-up number."""
    ni, dep, capex = f.get("net_income"), f.get("depreciation"), f.get("capex")
    if not _ok(ni, dep, capex):
        return na("owner earnings needs NI, D&A, capex")
    wc = None
    if _ok(f.get("current_assets"), f.get("current_liabilities"),
           f.get("current_assets_prev"), f.get("current_liabilities_prev")):
        wc = (f["current_assets"] - f["current_liabilities"]) - (
            f["current_assets_prev"] - f["current_liabilities_prev"])
    dwc = wc if wc is not None else 0.0
    oe = ni + dep - abs(capex) - dwc
    if oe <= 0:
        return na("owner earnings non-positive")
    if not _ok(f.get("shares")) or f["shares"] <= 0:
        return na("missing share count")
    growth, req, mos, n = 0.05, cost_of_equity(f, rf, erp), 0.25, 5
    pv = sum(oe * (1 + growth) ** t / (1 + req) ** t for t in range(1, n + 1))
    tg = min(growth, 0.03)
    if req <= tg:
        return na("owner earnings: cost of equity too low vs terminal growth to solve")
    term = (oe * (1 + growth) ** n * (1 + tg)) / (req - tg) / (1 + req) ** n
    val = (pv + term) * (1 - mos)
    return {"value": val / f["shares"], "applicable": True, "reason": "ok",
            "details": {"owner_earnings": oe, "mos": mos, "ke": req,
                        "owner_earnings_yield": oe / f["market_cap"] if _ok(f.get("market_cap")) and f["market_cap"] > 0 else None}}


def solve_implied_growth(base: float, r: float, ev_target: float
                         ) -> tuple[float | None, float | None, str | None]:
    """Binary-search the fade-to-terminal growth rate g0 that prices a
    Gordon-fade DCF (base flow fading linearly to TERMINAL_G over YEARS) at
    ev_target. Factored out of reverse_dcf so anchors.py's own-history
    growth percentile can re-solve this at every past quarter (same
    solver, different base/ev_target) without duplicating the search.
    Returns (implied_growth, solver_residual_pct, None) on success, or
    (None, None, reason) if ev_target sits outside the solvable
    [-50%, +100%]/yr range."""
    def pv_at(g0: float) -> float:
        pv, flow = 0.0, base
        for t in range(1, YEARS + 1):
            g_t = g0 + (TERMINAL_G - g0) * (t - 1) / (YEARS - 1)
            flow *= 1 + g_t
            pv += flow / (1 + r) ** t
        return pv + flow * (1 + TERMINAL_G) / (r - TERMINAL_G) / (1 + r) ** YEARS

    lo, hi = -0.5, 1.0
    if pv_at(lo) > ev_target:
        return None, None, "price below the -50%/yr floor value (implied growth < -50%)"
    if pv_at(hi) < ev_target:
        return None, None, "implied growth > 100%/yr: not solvable"
    for _ in range(80):
        mid = (lo + hi) / 2
        if pv_at(mid) < ev_target:
            lo = mid
        else:
            hi = mid
    g = (lo + hi) / 2
    return g, abs(pv_at(g) - ev_target) / ev_target, None


def reverse_dcf(f: dict, rf: float = RF_DEFAULT, erp: float = ERP) -> dict:
    """Solve the growth rate the current market cap implies. Universal: never a
    point estimate, always a falsifiable claim. Base flow: FCF if positive,
    else NOPAT*(1-30% reinvestment) if positive, else inapplicable (report says
    market is pricing a margin turnaround we won't guess)."""
    mcap = f.get("market_cap")
    if not _ok(mcap) or mcap <= 0:
        return na("missing market cap")
    fcf, nopat = f.get("fcf"), nopat_of(f)
    if _ok(fcf) and fcf > 0:
        base, kind = fcf, "FCF"
    elif _ok(nopat) and nopat > 0:
        base, kind = nopat * 0.7, "NOPAT(70%)"
    else:
        return na("no positive cash/earnings base: market prices a turnaround")
    r = wacc(f, rf, erp)
    net_debt = (f.get("total_debt") or 0.0) - (f.get("cash") or 0.0)
    ev_target = mcap + net_debt
    g, resid, oor_reason = solve_implied_growth(base, r, ev_target)
    if g is None:
        return na(oor_reason)
    return {"value": None, "applicable": True, "reason": "ok",
            "details": {"implied_growth": g, "base_kind": kind, "wacc": r,
                        "solver_residual": resid, "terminal_g": TERMINAL_G,
                        "street_growth_est": f.get("growth_est"),
                        "hist_growth_5y": f.get("revenue_growth_5y")}}


# ---------------------------------------------------------------- quality overlay

def accruals_check(f: dict) -> dict:
    ni, cfo, ta = f.get("net_income"), f.get("cfo"), f.get("total_assets")
    if not _ok(ni, cfo, ta) or ta <= 0:
        return na("accruals needs NI, CFO, assets")
    ratio = (ni - cfo) / ta
    return {"value": ratio, "applicable": True,
            "reason": "ok", "details": {"flag_high_accruals": ratio > 0.10}}


def piotroski_f(f: dict) -> dict:
    """9 binary signals (Piotroski 2000). Skips (and names) signals whose data
    is missing rather than scoring them 0."""
    checks, missing = {}, []

    def sig(name, cond_fields, cond):
        if not _ok(*cond_fields):
            missing.append(name)
        else:
            checks[name] = bool(cond())

    ta, tap = f.get("total_assets"), f.get("total_assets_prev")
    ni, nip = f.get("net_income"), f.get("net_income_prev")
    sig("roa_pos", (ni, ta), lambda: ni / ta > 0)
    sig("cfo_pos", (f.get("cfo"),), lambda: f["cfo"] > 0)
    sig("roa_up", (ni, ta, nip, tap), lambda: ni / ta > nip / tap)
    sig("cfo_gt_ni", (f.get("cfo"), ni), lambda: f["cfo"] > ni)
    sig("lev_down", (f.get("ltd"), ta, f.get("ltd_prev"), tap),
        lambda: f["ltd"] / ta <= f["ltd_prev"] / tap)
    sig("liq_up", (f.get("current_assets"), f.get("current_liabilities"),
                   f.get("current_assets_prev"), f.get("current_liabilities_prev")),
        lambda: f["current_assets"] / f["current_liabilities"]
        > f["current_assets_prev"] / f["current_liabilities_prev"])
    sig("no_dilution", (f.get("shares"), f.get("shares_prev")),
        lambda: f["shares"] <= f["shares_prev"] * 1.02)
    sig("margin_up", (f.get("gross_profit"), f.get("revenue"),
                      f.get("gross_profit_prev"), f.get("revenue_prev")),
        lambda: f["gross_profit"] / f["revenue"] > f["gross_profit_prev"] / f["revenue_prev"])
    sig("turnover_up", (f.get("revenue"), ta, f.get("revenue_prev"), tap),
        lambda: f["revenue"] / ta > f["revenue_prev"] / tap)
    if len(checks) < 5:
        return na(f"too few F-score inputs (missing: {', '.join(missing)})")
    score = sum(checks.values())
    return {"value": score, "applicable": True, "reason": "ok",
            "details": {"of": len(checks), "signals": checks, "missing": missing}}


def mohanram_g(f: dict) -> dict:
    """Adapted G-score (Mohanram 2005 uses industry medians we don't have;
    adaptation: absolute profitability + own-history stability + investment
    intensity vs assets). Explicitly labeled 'adapted' in reports."""
    checks, missing = {}, []
    ta = f.get("total_assets")
    if not _ok(ta) or ta <= 0:
        return na("G-score needs total assets")

    def sig(name, cond_fields, cond):
        if not _ok(*cond_fields):
            missing.append(name)
        else:
            checks[name] = bool(cond())

    sig("roa_pos", (f.get("net_income"),), lambda: f["net_income"] / ta > 0)
    sig("cfo_roa_pos", (f.get("cfo"),), lambda: f["cfo"] / ta > 0)
    sig("cfo_gt_ni", (f.get("cfo"), f.get("net_income")), lambda: f["cfo"] > f["net_income"])
    sig("rev_growing", (f.get("revenue"), f.get("revenue_prev")),
        lambda: f["revenue"] > f["revenue_prev"])
    sig("rnd_intense", (f.get("rnd"),), lambda: f["rnd"] / ta > 0.02)
    sig("capex_intense", (f.get("capex"),), lambda: abs(f["capex"]) / ta > 0.03)
    sig("margin_stable", (f.get("gross_profit"), f.get("revenue"),
                          f.get("gross_profit_prev"), f.get("revenue_prev")),
        lambda: abs(f["gross_profit"] / f["revenue"]
                    - f["gross_profit_prev"] / f["revenue_prev"]) < 0.05)
    sig("no_dilution", (f.get("shares"), f.get("shares_prev")),
        lambda: f["shares"] <= f["shares_prev"] * 1.02)
    if len(checks) < 4:
        return na(f"too few G-score inputs (missing: {', '.join(missing)})")
    return {"value": sum(checks.values()), "applicable": True, "reason": "ok",
            "details": {"of": len(checks), "signals": checks, "missing": missing,
                        "adapted": True}}


def altman_z(f: dict) -> dict:
    ta, tl = f.get("total_assets"), f.get("total_liabilities")
    if not _ok(ta, tl, f.get("current_assets"), f.get("current_liabilities"),
               f.get("retained_earnings"), f.get("ebit"), f.get("market_cap"),
               f.get("revenue")) or ta <= 0 or tl <= 0:
        return na("Z-score needs full balance sheet + EBIT + market cap")
    z = (1.2 * (f["current_assets"] - f["current_liabilities"]) / ta
         + 1.4 * f["retained_earnings"] / ta
         + 3.3 * f["ebit"] / ta
         + 0.6 * f["market_cap"] / tl
         + 1.0 * f["revenue"] / ta)
    zone = "safe" if z > 2.99 else ("grey" if z >= 1.81 else "distress")
    return {"value": z, "applicable": True, "reason": "ok", "details": {"z": z, "zone": zone}}


def roce_decomposition(f: dict) -> dict:
    """Penman ROCE decomposition: RNOA = PM x ATO; ROCE = RNOA + FLEV x SPREAD,
    SPREAD = RNOA - NBC. Surfaces the leverage caveat directly (notes: leverage
    lifts ROCE only when SPREAD > 0; otherwise reported growth from debt is not
    value creation, just a redistribution the LLM personas should discount)."""
    nopat, noa = nopat_of(f), noa_of(f)
    revenue, equity = f.get("revenue"), f.get("equity")
    if not _ok(nopat, noa, revenue) or noa <= 0 or revenue <= 0:
        return na("ROCE decomposition needs EBIT, NOA and revenue")
    rnoa, pm, ato = nopat / noa, nopat / revenue, revenue / noa
    details = {"rnoa": rnoa, "pm": pm, "ato": ato}

    ni = f.get("net_income")
    if _ok(ni, equity) and equity > 0:
        details["roce"] = ni / equity
    if _ok(equity) and equity > 0:
        nfo = (f.get("total_debt") or 0.0) - (f.get("cash") or 0.0)
        flev = nfo / equity
        details["flev"] = flev
        interest = f.get("interest_expense")
        if _ok(interest) and nfo != 0:
            nbc = abs(interest) * (1 - effective_tax(f)) / nfo
            spread = rnoa - nbc
            details.update(nbc=nbc, spread=spread, roce_from_decomposition=rnoa + flev * spread,
                           leverage_amplifying_without_spread=flev > 0.1 and spread <= 0)
        else:
            details["spread_reason"] = "no interest-expense line item available"
    return {"value": rnoa, "applicable": True, "reason": "ok", "details": details}


def quick_ratios(f: dict, rf: float = RF_DEFAULT) -> dict:
    """Cheap textbook cross-checks. Both are explicitly rough heuristics per
    the source material, not standalone valuation models."""
    out: dict = {}
    price, eps1, g = f.get("price"), f.get("eps_fwd"), f.get("growth_est")
    if _ok(price, eps1) and eps1 > 0 and _ok(g) and g > 0:
        pe = price / eps1
        out["peg"] = {"value": pe / (g * 100),
                      "details": {"pe_fwd": pe, "growth_pct": g * 100},
                      "note": "rough heuristic: <1 often read bullish, >1 bearish; "
                              "assumes constant growth"}
    if _ok(price, eps1) and price > 0:
        ey = eps1 / price
        out["fed_model"] = {"earnings_yield": ey, "ten_year_yield": rf, "spread": ey - rf,
                            "note": "earnings yield below the 10y yield is the classic "
                                    "'stocks rich vs bonds' Fed-model read; a very rough "
                                    "heuristic, not a valuation model"}
    return out


def beneish_m(f: dict) -> dict:
    """Manipulation veto flag (M > -1.78 trips). Needs 2 years of detail
    INCLUDING net_ppe (AQI is meaningless with PPE silently treated as 0);
    skips with named reason when fields are missing (low coverage is expected)."""
    req = ("receivables", "receivables_prev", "revenue", "revenue_prev",
           "gross_profit", "gross_profit_prev", "total_assets", "total_assets_prev",
           "depreciation", "depreciation_prev", "sga", "sga_prev", "ltd", "ltd_prev",
           "current_liabilities", "current_liabilities_prev", "cfo", "net_income",
           "current_assets", "current_assets_prev", "net_ppe", "net_ppe_prev")
    if not _ok(*(f.get(k) for k in req)):
        return na("Beneish M needs 2y statement detail incl. net_ppe (often unavailable)")
    dsri = (f["receivables"] / f["revenue"]) / (f["receivables_prev"] / f["revenue_prev"])
    gmi = (f["gross_profit_prev"] / f["revenue_prev"]) / (f["gross_profit"] / f["revenue"])
    aqi_t = 1 - (f["current_assets"] + f["net_ppe"]) / f["total_assets"]
    aqi_p = 1 - (f["current_assets_prev"] + f["net_ppe_prev"]) / f["total_assets_prev"]
    aqi = aqi_t / aqi_p if aqi_p else 1.0
    sgi = f["revenue"] / f["revenue_prev"]
    depi = ((f["depreciation_prev"] / (f["depreciation_prev"] + f["net_ppe_prev"]))
            / (f["depreciation"] / (f["depreciation"] + f["net_ppe"])))
    sgai = (f["sga"] / f["revenue"]) / (f["sga_prev"] / f["revenue_prev"])
    lvgi = ((f["ltd"] + f["current_liabilities"]) / f["total_assets"]) / (
        (f["ltd_prev"] + f["current_liabilities_prev"]) / f["total_assets_prev"])
    tata = (f["net_income"] - f["cfo"]) / f["total_assets"]
    m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    return {"value": m, "applicable": True, "reason": "ok",
            "details": {"flag_manipulation": m > -1.78}}


def core_ebit_check(f: dict) -> dict:
    """Yahoo's EBIT can include one-off items (restructuring, impairments,
    gains on sale) that distort ReOI's growth driver. Surfaces core vs
    reported EBIT as a data-quality flag rather than silently swapping the
    models' own EBIT input, which stays a separate decision."""
    ebit, unusual = f.get("ebit"), f.get("ebit_unusual_items")
    if not _ok(ebit, unusual) or ebit == 0:
        return na("no 'Total Unusual Items' line item available from statements")
    core_ebit = ebit - unusual
    pct = abs(unusual) / abs(ebit)
    return {"value": pct, "applicable": True, "reason": "ok",
            "details": {"ebit_reported": ebit, "ebit_core": core_ebit,
                        "unusual_items": unusual, "flag_material_unusual_items": pct > 0.10}}


def buyback_value_check(f: dict, fair_value: float | None) -> dict:
    """Textbook caveat (notes sec. 6.3): buybacks redistribute value, they
    don't create it, UNLESS done below fair value. Implied avg repurchase
    price = buyback $ / net share-count reduction, compared against the
    model's fair value. Heuristic: share-count change also reflects SBC
    issuance, so this is a rough signal, not a precise number."""
    buyback_cash, shares, shares_prev = (
        f.get("buybacks"), f.get("shares"), f.get("shares_prev"))
    if not _ok(buyback_cash, shares, shares_prev) or buyback_cash <= 0:
        return na("no material buyback cash outflow or share-count data")
    share_reduction = shares_prev - shares
    if share_reduction <= 0:
        return na("share count did not shrink net of buybacks (issuance offset or grew)")
    implied_price = buyback_cash / share_reduction
    details = {"implied_avg_price": implied_price, "buyback_cash": buyback_cash,
               "shares_reduced": share_reduction}
    if _ok(fair_value):
        pct = (implied_price - fair_value) / fair_value
        details.update(fair_value=fair_value, premium_to_fair_value_pct=pct * 100,
                       flag_buybacks_above_fair_value=pct > 0.10)
    else:
        details["note"] = "no fair value available yet to compare against"
    return {"value": implied_price, "applicable": True, "reason": "ok", "details": details}


def cash_conversion_cycle(f: dict) -> dict:
    """DSO + DIO - DPO (notes sec. 4.3 turnover drivers). Uses period-end
    balances (not averages) since only the latest two annual columns are
    fetched; DIO/DPO are each individually omittable when inventory/payables
    don't apply (e.g. software, some financials) rather than failing whole."""
    revenue, receivables, gross_profit = (
        f.get("revenue"), f.get("receivables"), f.get("gross_profit"))
    if not _ok(revenue, receivables, gross_profit) or revenue <= 0:
        return na("cash conversion cycle needs revenue, receivables, gross profit")
    cogs = revenue - gross_profit
    details = {"dso": receivables / revenue * 365}
    inventory, payables = f.get("inventory"), f.get("payables")
    if _ok(inventory) and cogs > 0:
        details["dio"] = inventory / cogs * 365
    if _ok(payables) and cogs > 0:
        details["dpo"] = payables / cogs * 365
    if "dio" in details and "dpo" in details:
        details["cash_conversion_cycle"] = details["dso"] + details["dio"] - details["dpo"]
    return {"value": details.get("cash_conversion_cycle", details["dso"]),
            "applicable": True, "reason": "ok", "details": details}


def data_coverage_score(f: dict, expected_fields: tuple[str, ...]) -> dict:
    """Fraction of the named fundamental fields that are actually populated.
    A fair value built on 90% coverage deserves more trust than one built on
    60% — this makes that difference visible instead of implicit."""
    present = sum(1 for k in expected_fields if _ok(f.get(k)))
    return {"value": present / len(expected_fields), "applicable": True, "reason": "ok",
            "details": {"present": present, "of": len(expected_fields),
                        "missing": [k for k in expected_fields if not _ok(f.get(k))]}}


DATA_COVERAGE_FIELDS = (
    "revenue", "ebit", "net_income", "pretax_income", "tax_provision", "gross_profit",
    "cfo", "capex", "depreciation", "fcf", "total_assets", "total_liabilities",
    "current_assets", "current_liabilities", "cash", "total_debt", "equity",
    "retained_earnings", "receivables", "shares", "market_cap", "price", "beta",
    "eps_fwd", "growth_est", "revenue_growth_5y", "interest_expense",
)


def analyst_target_triangulation(f: dict) -> dict:
    """Surfaces the Street's consensus price target next to the model blend
    for context only — never folded into fair_value, so it stays a check,
    not a crutch that lets the model blend hide behind consensus."""
    target, price = f.get("analyst_target_mean"), f.get("price")
    if not _ok(target) or target <= 0:
        return na("no analyst price target available")
    details = {"analyst_target_mean": target, "analyst_count": f.get("analyst_target_count")}
    if _ok(price) and price > 0:
        details["target_vs_price_pct"] = (target - price) / price * 100
    return {"value": target, "applicable": True, "reason": "ok", "details": details}


# ---------------------------------------------------------------- industry drivers

# Textbook (Penman ch.8) key-ReOI-driver map by industry keyword. Purely a
# prompt hint for the LLM personas ("focus on the 1-2 drivers that matter for
# this business" per the forecasting-discipline notes) — never fed into any
# deterministic model, so a missed match just means no hint, never a wrong number.
INDUSTRY_DRIVERS = (
    ("airline", "Load factors and fares -> watch Sales and ATO."),
    ("auto", "Model design and production efficiency -> watch Sales and margins."),
    ("beverage", "Brand management and product innovation -> watch Sales."),
    ("wireless", "Population covered and churn rates -> watch Sales and ATO."),
    ("telecom", "Population covered and churn rates -> watch Sales and ATO."),
    ("real estate", "Square footage and occupancy rates -> watch Sales and ATO."),
    ("reit", "Square footage and occupancy rates -> watch Sales and ATO."),
    ("hardware", "Technology path and competition -> watch Sales and margins."),
    ("semiconductor", "Technology path and competition -> watch Sales and margins."),
    ("apparel", "Brand management -> watch Sales and margins."),
    ("internet", "Traffic/engagement -> watch Sales and ATO."),
    ("retail", "Retail space and sales per square foot -> watch Sales and ATO."),
    ("pharma", "Research and development pipeline -> watch Sales."),
    ("biotechnology", "Research and development pipeline -> watch Sales."),
)


def industry_driver_hint(industry: str | None) -> str | None:
    """Best-effort keyword match; returns None (never a guess) when the
    industry string doesn't match a known keyword."""
    if not industry:
        return None
    low = industry.lower()
    return next((hint for kw, hint in INDUSTRY_DRIVERS if kw in low), None)


# ---------------------------------------------------------------- assembly

REGIME_MODELS = {
    "mature-profitable": ("reoi", "reverse_dcf", "aeg", "owner_earnings"),
    "financial": ("equity_rim", "reverse_dcf", "aeg"),
    "growth-unprofitable": ("reverse_dcf", "aeg"),
    "distressed": ("reverse_dcf",),
    "etf": (),
    "unsupported": (),
    "unknown": (),
}
MODEL_FN = {
    "reoi": reoi_value, "equity_rim": equity_rim_value, "aeg": aeg_value,
    "owner_earnings": owner_earnings_value, "reverse_dcf": reverse_dcf,
}
# Ensemble weights among point-estimate models (reverse_dcf has no point value).
# NOTE: chosen, not fit. Tier 2 calibration (calibrate_weights_from_backtest)
# exists to replace these with measured inverse-error weights once a real
# historical backtest has been run — see backtest.py. Do not hand-edit these
# without backtest evidence; that would swap one guess for another.
WEIGHTS = {"reoi": 0.5, "equity_rim": 0.5, "aeg": 0.3, "owner_earnings": 0.2}


def monte_carlo_range(f: dict, rf: float = RF_DEFAULT, erp: float = ERP,
                      n: int = 500, seed: int | None = 7) -> dict:
    """Tier 3: a real confidence band instead of one dollar point estimate.
    Jointly jitters beta, rf, erp, the ReOI/RIM competitive-fade floor, and
    terminal growth (independent uniform perturbations per sample) and
    reports percentiles of the weighted fair value. AEG/reverse-DCF/owner-
    earnings keep their own fixed assumptions in this pass (not yet
    parameterized) — the band is a lower bound on true uncertainty, not the
    full picture."""
    rng = random.Random(seed)
    base_beta = f.get("beta") if _ok(f.get("beta")) else 1.0
    samples = []
    regime = classify_regime(f)["regime"]
    names = [n for n in REGIME_MODELS[regime] if n in WEIGHTS]
    for _ in range(n):
        f_j = dict(f, beta=base_beta * rng.uniform(0.85, 1.15))
        rf_j = rf * rng.uniform(0.85, 1.15)
        erp_j = erp * rng.uniform(0.85, 1.15)
        fade_j = rng.uniform(0.15, 0.35)
        tg_j = TERMINAL_G * rng.uniform(0.6, 1.4)
        pts = []
        for name in names:
            fn = MODEL_FN[name]
            m = (fn(f_j, rf_j, erp_j, fade_floor=fade_j, terminal_g=tg_j)
                 if name in ("reoi", "equity_rim") else fn(f_j, rf_j, erp_j))
            if m["applicable"] and m["value"] is not None:
                pts.append((WEIGHTS[name], m["value"]))
        if pts:
            tw = sum(w for w, _ in pts)
            samples.append(sum(w * v for w, v in pts) / tw)
    if not samples:
        return na("no applicable point-estimate models to simulate")
    samples.sort()

    def pct(p: float) -> float:
        idx = min(len(samples) - 1, max(0, round(p * (len(samples) - 1))))
        return samples[idx]

    return {"value": pct(0.5), "applicable": True, "reason": "ok",
            "details": {"p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90), "n": len(samples)}}


def run_valuation(f: dict, rf: float = RF_DEFAULT, erp: float = ERP,
                  rf_source: str | None = None, erp_source: str | None = None) -> dict:
    regime = classify_regime(f)
    out = {"regime": regime, "models": {}, "quality": {}, "fair_value": None,
           "upside_pct": None, "sensitivity": None,
           "rates_used": {"rf": rf, "erp": erp, "rf_source": rf_source, "erp_source": erp_source}}
    for name in REGIME_MODELS[regime["regime"]]:
        out["models"][name] = MODEL_FN[name](f, rf, erp)
    # weighted fair value across applicable point-estimate models
    pts = [(WEIGHTS[n], m["value"]) for n, m in out["models"].items()
           if m["applicable"] and m["value"] is not None and n in WEIGHTS]
    if pts:
        tw = sum(w for w, _ in pts)
        out["fair_value"] = sum(w * v for w, v in pts) / tw
        if _ok(f.get("price")) and f["price"] > 0:
            out["upside_pct"] = (out["fair_value"] - f["price"]) / f["price"] * 100
        # sensitivity: rf +/-1% shifts every discount rate; report the band
        vals = []
        for shift in (-0.01, 0.01):
            p = [(WEIGHTS[n], MODEL_FN[n](f, rf + shift, erp)["value"])
                 for n in out["models"] if n in WEIGHTS]
            p = [(w, v) for w, v in p if v is not None]
            if p:
                vals.append(sum(w * v for w, v in p) / sum(w for w, _ in p))
        if len(vals) == 2:
            out["sensitivity"] = {"rate_minus_1pct": max(vals), "rate_plus_1pct": min(vals)}
    for qname, qfn in (("accruals", accruals_check), ("piotroski_f", piotroski_f),
                       ("mohanram_g", mohanram_g), ("altman_z", altman_z),
                       ("beneish_m", beneish_m), ("core_ebit", core_ebit_check),
                       ("cash_conversion_cycle", cash_conversion_cycle)):
        out["quality"][qname] = qfn(f)
    out["quality"]["buyback_value_check"] = buyback_value_check(f, out["fair_value"])
    out["roce_decomposition"] = roce_decomposition(f)
    out["quick_ratios"] = quick_ratios(f, rf)
    out["data_coverage"] = data_coverage_score(f, DATA_COVERAGE_FIELDS)
    out["analyst_target"] = analyst_target_triangulation(f)
    out["confidence_range"] = monte_carlo_range(f, rf, erp) if pts else na("no fair value to simulate")
    return out

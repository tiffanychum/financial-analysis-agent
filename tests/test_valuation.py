"""Golden-number and invariant tests for the deterministic engine.

Layer 1 (formulas): synthetic fixtures with hand-computed expected values.
Layer 2 (invariants): identities that must hold regardless of inputs
(reverse-DCF residual ~ 0, ReOI collapses to NOA when RNOA == WACC,
regime routing excludes models by rule, never silent-zero).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import valuation as V  # noqa: E402


def base_firm(**over) -> dict:
    """Mature profitable firm with round numbers; beta 1.0 -> ke = rf + ERP."""
    f = dict(
        quote_type="EQUITY", sector="Technology",
        # prev-year values chosen so ALL 9 Piotroski signals pass:
        # roa up (0.125 > 0.109), turnover up (0.833 > 0.818),
        # margin up (0.500 > 0.489), liquidity up (2.00 > 1.95)
        revenue=1000.0, revenue_prev=900.0,
        ebit=200.0, pretax_income=190.0, tax_provision=190.0 * 0.25,
        net_income=150.0, net_income_prev=120.0,
        gross_profit=500.0, gross_profit_prev=440.0,
        cfo=180.0, cfo_prev=170.0, capex=-50.0, depreciation=40.0,
        total_assets=1200.0, total_assets_prev=1100.0,
        total_liabilities=500.0,
        current_assets=400.0, current_assets_prev=380.0,
        current_liabilities=200.0, current_liabilities_prev=195.0,
        cash=100.0, total_debt=150.0, total_debt_prev=160.0,
        ltd=120.0, ltd_prev=130.0,
        equity=700.0, retained_earnings=400.0,
        receivables=80.0, receivables_prev=76.0,
        sga=150.0, sga_prev=145.0, rnd=50.0,
        shares=100.0, shares_prev=100.0,
        market_cap=3000.0, price=30.0, beta=1.0,
        fcf=130.0, eps_fwd=1.6, growth_est=0.08,
        revenue_growth_5y=0.05,
    )
    f.update(over)
    return f


# ---------------------------------------------------------------- layer 1: golden

def test_effective_tax_golden():
    assert math.isclose(V.effective_tax(base_firm()), 0.25, rel_tol=1e-9)


def test_nopat_golden():
    # NOPAT = EBIT * (1 - tax) = 200 * 0.75 = 150
    assert math.isclose(V.nopat_of(base_firm()), 150.0, rel_tol=1e-9)


def test_noa_golden():
    # NOA = (1200 - 100) - (500 - 150) = 750
    assert math.isclose(V.noa_of(base_firm()), 750.0, rel_tol=1e-9)


def test_cost_of_equity_golden():
    # ke = 0.042 + 1.0 * 0.05 = 0.092
    assert math.isclose(V.cost_of_equity(base_firm()), 0.092, rel_tol=1e-9)


def test_cost_of_equity_uses_live_rf_and_erp():
    # ke = rf + beta * erp; live macro inputs must actually move the output,
    # not just fall back to the module constants.
    assert math.isclose(V.cost_of_equity(base_firm(), rf=0.03, erp=0.045), 0.075, rel_tol=1e-9)


def test_wacc_golden():
    # ke = 0.092; kd_at = 0.05*(1-0.25) = 0.0375; wd = 150/3150
    f = base_firm()
    wd = 150.0 / 3150.0
    expected = (1 - wd) * 0.092 + wd * 0.0375
    assert math.isclose(V.wacc(f), expected, rel_tol=1e-9)


def test_wacc_falls_back_to_ke_with_no_debt():
    f = base_firm(total_debt=0.0)
    assert math.isclose(V.wacc(f), V.cost_of_equity(f), rel_tol=1e-9)


def test_roce_decomposition_golden():
    """RNOA = NOPAT/NOA = 150/750 = 0.20; PM = NOPAT/Rev = 150/1000 = 0.15;
    ATO = Rev/NOA = 1000/750; FLEV = NFO/CSE = (150-100)/700;
    NBC = |interest|*(1-tax)/NFO = 10*0.75/50 = 0.15; SPREAD = 0.20-0.15 = 0.05."""
    f = base_firm(interest_expense=10.0)
    got = V.roce_decomposition(f)
    assert got["applicable"]
    d = got["details"]
    assert math.isclose(d["rnoa"], 0.20, rel_tol=1e-9)
    assert math.isclose(d["pm"], 0.15, rel_tol=1e-9)
    assert math.isclose(d["ato"], 1000 / 750, rel_tol=1e-9)
    assert math.isclose(d["flev"], 50 / 700, rel_tol=1e-9)
    assert math.isclose(d["nbc"], 0.15, rel_tol=1e-9)
    assert math.isclose(d["spread"], 0.05, rel_tol=1e-9)
    assert math.isclose(d["roce_from_decomposition"], d["rnoa"] + d["flev"] * d["spread"], rel_tol=1e-9)
    assert d["leverage_amplifying_without_spread"] is False


def test_roce_decomposition_missing_interest_expense_omits_spread():
    got = V.roce_decomposition(base_firm())
    assert got["applicable"]
    assert "spread" not in got["details"]
    assert "spread_reason" in got["details"]


def test_roce_decomposition_flags_leverage_without_spread():
    """Same RNOA but NBC above RNOA: leverage is book-boosting ROCE without
    creating value — the textbook's core caveat about financial leverage."""
    # nfo = 250-100 = 150 -> flev = 150/700 = 0.214 (>0.1); nbc = 60*0.75/150 = 0.30 > rnoa 0.20
    f = base_firm(total_debt=250.0, interest_expense=60.0)
    got = V.roce_decomposition(f)
    assert got["details"]["leverage_amplifying_without_spread"] is True


def test_quick_ratios_peg_and_fed_model_golden():
    f = base_firm()  # price 30, eps_fwd 1.6, growth_est 0.08, rf default 0.042
    out = V.quick_ratios(f, rf=0.042)
    pe = 30.0 / 1.6
    assert math.isclose(out["peg"]["value"], pe / 8.0, rel_tol=1e-9)
    ey = 1.6 / 30.0
    assert math.isclose(out["fed_model"]["earnings_yield"], ey, rel_tol=1e-9)
    assert math.isclose(out["fed_model"]["spread"], ey - 0.042, rel_tol=1e-9)


def test_owner_earnings_golden():
    """Hand computation with the ai-hedge-fund parametrization (g=5%,
    MOS=25%, 5y + Gordon terminal at 3%), discounted at the SAME live
    cost-of-equity every other model uses (rf + beta*erp = 0.092 for
    base_firm, not a fixed 15% hurdle):
    OE = 150 + 40 - 50 - dWC;  dWC = (400-200) - (380-195) = 15  -> OE = 125."""
    f = base_firm()
    oe, r = 125.0, 0.092
    pv = sum(oe * 1.05 ** t / (1 + r) ** t for t in range(1, 6))
    term = (oe * 1.05 ** 5 * 1.03) / (r - 0.03) / (1 + r) ** 5
    expected = (pv + term) * 0.75 / 100.0
    got = V.owner_earnings_value(f)
    assert got["applicable"]
    assert math.isclose(got["value"], expected, rel_tol=1e-9)
    assert math.isclose(got["details"]["ke"], r, rel_tol=1e-9)


def test_owner_earnings_yield_computed_when_market_cap_present():
    f = base_firm()
    got = V.owner_earnings_value(f)
    assert math.isclose(got["details"]["owner_earnings_yield"], 125.0 / f["market_cap"], rel_tol=1e-9)


def test_owner_earnings_uses_live_rf_and_erp_not_fixed_hurdle():
    """The old bug: req=0.15 was hardcoded, so passing a different rf/erp did
    nothing. This pins that owner_earnings_value output actually moves."""
    f = base_firm()
    default = V.owner_earnings_value(f)
    shifted = V.owner_earnings_value(f, rf=0.08, erp=0.08)
    assert shifted["value"] != default["value"]
    assert math.isclose(shifted["details"]["ke"], 0.08 + 1.0 * 0.08, rel_tol=1e-9)


def test_mohanram_g_golden_8_of_8():
    # base_firm is engineered so all 8 adapted-G signals pass (see docstring math)
    got = V.mohanram_g(base_firm())
    assert got["applicable"]
    assert got["details"]["of"] == 8
    assert got["value"] == 8
    assert got["details"]["adapted"] is True


def test_beneish_m_golden_independent_recompute():
    """Independent recompute of the 8-variable index (not copied from the
    function body) on a fixture with full 2y detail incl. net_ppe."""
    f = base_firm(net_ppe=300.0, net_ppe_prev=290.0, depreciation_prev=38.0,
                  interest_expense=10.0)
    dsri = (80.0 / 1000.0) / (76.0 / 900.0)
    gmi = (440.0 / 900.0) / (500.0 / 1000.0)
    aqi_t = 1 - (400.0 + 300.0) / 1200.0
    aqi_p = 1 - (380.0 + 290.0) / 1100.0
    aqi = aqi_t / aqi_p
    sgi = 1000.0 / 900.0
    depi = (38.0 / (38.0 + 290.0)) / (40.0 / (40.0 + 300.0))
    sgai = (150.0 / 1000.0) / (145.0 / 900.0)
    lvgi = ((120.0 + 200.0) / 1200.0) / ((130.0 + 195.0) / 1100.0)
    tata = (150.0 - 180.0) / 1200.0
    expected = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
                + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    got = V.beneish_m(f)
    assert got["applicable"]
    assert math.isclose(got["value"], expected, rel_tol=1e-9)
    assert got["details"]["flag_manipulation"] is False


def test_beneish_m_excludes_without_net_ppe():
    """Tier 1 fix: AQI is meaningless with PPE silently defaulted to 0, so
    net_ppe is now required, not optional."""
    f = base_firm(depreciation_prev=38.0)  # no net_ppe / net_ppe_prev
    got = V.beneish_m(f)
    assert got["applicable"] is False
    assert "net_ppe" in got["reason"]


def test_altman_z_golden():
    """Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MC/TL + 1.0*REV/TA."""
    f = base_firm()
    expected = (1.2 * 200 / 1200 + 1.4 * 400 / 1200 + 3.3 * 200 / 1200
                + 0.6 * 3000 / 500 + 1.0 * 1000 / 1200)
    got = V.altman_z(f)
    assert got["applicable"]
    assert math.isclose(got["value"], expected, rel_tol=1e-9)
    assert got["details"]["zone"] == "safe"


def test_piotroski_golden_9_of_9():
    # base_firm is engineered so all 9 signals pass
    got = V.piotroski_f(base_firm())
    assert got["applicable"]
    assert got["details"]["of"] == 9
    assert got["value"] == 9


def test_piotroski_signal_flips():
    # dilution: shares grew >2% -> exactly one signal flips off
    got = V.piotroski_f(base_firm(shares=105.0, shares_prev=100.0))
    assert got["value"] == 8
    assert got["details"]["signals"]["no_dilution"] is False


def test_accruals_golden():
    # (NI - CFO) / TA = (150 - 180) / 1200 = -0.025 (negative = high quality)
    got = V.accruals_check(base_firm())
    assert math.isclose(got["value"], -0.025, rel_tol=1e-9)
    assert got["details"]["flag_high_accruals"] is False


# ---------------------------------------------------------------- layer 2: invariants

def test_reoi_collapses_to_noa_when_spread_zero():
    """RNOA == WACC -> zero value creation -> EV == NOA exactly."""
    f = base_firm()
    r = V.nopat_of(f) / V.noa_of(f)  # force wacc = rnoa
    got = V.reoi_value(f, w=r)
    assert got["applicable"]
    net_debt = f["total_debt"] - f["cash"]
    expected = (V.noa_of(f) - net_debt) / f["shares"]
    assert math.isclose(got["value"], expected, rel_tol=1e-9)


def test_reoi_nonzero_spread_matches_forward_construction():
    """Independent forward construction of the exact fade/terminal formula
    the docstring claims, for a nonzero RNOA-WACC spread (previously only the
    zero-spread identity was pinned)."""
    f = base_firm()
    r = V.wacc(f)
    nopat, noa = V.nopat_of(f), V.noa_of(f)
    rnoa0 = nopat / noa
    spread0 = rnoa0 - r
    g = min(f["revenue_growth_5y"], 0.12)
    pv, noa_t, reoi_t = 0.0, noa, 0.0
    for t in range(1, V.YEARS + 1):
        fade = 1 - 0.75 * (t - 1) / (V.YEARS - 1)
        reoi_t = spread0 * fade * noa_t
        pv += reoi_t / (1 + r) ** t
        noa_t *= 1 + g * (1 - (t - 1) / V.YEARS)
    term = reoi_t * (1 + V.TERMINAL_G) / (r - V.TERMINAL_G) / (1 + r) ** V.YEARS
    ev = noa + pv + term
    expected = (ev - (f["total_debt"] - f["cash"])) / f["shares"]
    got = V.reoi_value(f)
    assert got["applicable"]
    assert math.isclose(got["value"], expected, rel_tol=1e-9)


def test_reoi_fade_floor_and_terminal_g_are_threaded():
    """Params exist so monte_carlo_range() can perturb them per-sample, not
    just read the module constants."""
    f = base_firm()
    default = V.reoi_value(f)
    shifted = V.reoi_value(f, fade_floor=0.6, terminal_g=0.01)
    assert shifted["value"] != default["value"]


def test_equity_rim_nonzero_spread_matches_forward_construction():
    f = base_firm(sector="Financial Services")
    ke = V.cost_of_equity(f)
    ni, bv = f["net_income"], f["equity"]
    roe0 = ni / bv
    spread0 = roe0 - ke
    payout = 0.4
    pv, bv_t, ri_t = 0.0, bv, 0.0
    for t in range(1, V.YEARS + 1):
        fade = 1 - 0.75 * (t - 1) / (V.YEARS - 1)
        roe_t = ke + spread0 * fade
        ni_t = roe_t * bv_t
        ri_t = ni_t - ke * bv_t
        pv += ri_t / (1 + ke) ** t
        bv_t += ni_t * (1 - payout)
    term = ri_t * (1 + V.TERMINAL_G) / (ke - V.TERMINAL_G) / (1 + ke) ** V.YEARS
    expected = (bv + pv + term) / f["shares"]
    got = V.equity_rim_value(f)
    assert got["applicable"]
    assert math.isclose(got["value"], expected, rel_tol=1e-9)


def test_reverse_dcf_exclusion_copy_matches_actual_bound():
    """Tier 1 fix: message used to say 'zero-growth' while the real floor is
    -50%/yr; assert the corrected wording, not the old bug."""
    f = base_firm(market_cap=1.0)  # absurdly cheap -> below the -50% floor value
    got = V.reverse_dcf(f)
    assert got["applicable"] is False
    assert "-50%" in got["reason"]
    assert "zero-growth" not in got["reason"]


def test_reverse_dcf_solver_residual():
    got = V.reverse_dcf(base_firm())
    assert got["applicable"]
    assert got["details"]["solver_residual"] < 1e-3
    assert -0.5 < got["details"]["implied_growth"] < 1.0


def test_reverse_dcf_matches_forward_construction():
    """Build a firm priced exactly at pv_at(g=6%); solver must recover ~6%.
    Zero debt so WACC == ke and does not shift with the constructed market cap."""
    f = base_firm(total_debt=0.0)
    r, base, g0 = V.wacc(f), f["fcf"], 0.06
    pv, flow = 0.0, base
    for t in range(1, V.YEARS + 1):
        g_t = g0 + (V.TERMINAL_G - g0) * (t - 1) / (V.YEARS - 1)
        flow *= 1 + g_t
        pv += flow / (1 + r) ** t
    pv += flow * (1 + V.TERMINAL_G) / (r - V.TERMINAL_G) / (1 + r) ** V.YEARS
    net_debt = f["total_debt"] - f["cash"]
    f2 = base_firm(total_debt=0.0, market_cap=pv - net_debt)
    got = V.reverse_dcf(f2)
    assert abs(got["details"]["implied_growth"] - g0) < 1e-3


def test_aeg_zero_abnormal_growth_collapses_to_eps_over_r():
    """g == r with full payout -> AEG_t == 0 -> V = EPS1 / r.
    With payout p, AEG_t = eps_prev*(g - r + r*p); zero when g = r*(1-p)."""
    f = base_firm()
    r = V.cost_of_equity(f)
    g_star = r * (1 - 0.3)
    # freeze fade by setting growth_est = g_star and TERMINAL_G ~ g_star via monkeypatch
    old_tg = V.TERMINAL_G
    V.TERMINAL_G = g_star
    try:
        got = V.aeg_value(base_firm(growth_est=g_star))
        assert math.isclose(got["value"], f["eps_fwd"] / r, rel_tol=1e-6)
    finally:
        V.TERMINAL_G = old_tg


def test_regime_routing_negative_nopat_excludes_point_models():
    f = base_firm(ebit=-50.0, net_income=-40.0, fcf=-30.0)
    reg = V.classify_regime(f)
    assert reg["regime"] == "growth-unprofitable"
    out = V.run_valuation(f)
    assert set(out["models"]) == {"reverse_dcf", "aeg"}
    assert "owner_earnings" not in out["models"]
    # negative-FCF, negative-NOPAT firm: reverse dcf declares turnaround pricing
    assert out["models"]["reverse_dcf"]["applicable"] is False
    assert "turnaround" in out["models"]["reverse_dcf"]["reason"]


def test_regime_etf_skips_fundamentals():
    out = V.run_valuation({"quote_type": "ETF"})
    assert out["regime"]["regime"] == "etf"
    assert out["models"] == {}
    assert out["fair_value"] is None


def test_regime_unknown_when_quote_type_and_statements_both_missing():
    """quoteSummary blocked (quote_type None) + no statements resolved used
    to default all the way to 'mature-profitable' on zero evidence. Must
    land on 'unknown', not a going-concern label the committee argues
    about."""
    out = V.run_valuation({"quote_type": None, "price": 89.37, "beta": 0.28})
    assert out["regime"]["regime"] == "unknown"
    assert out["models"] == {}


def test_regime_unknown_when_quote_type_claims_equity_but_no_statements_exist():
    """Live-checked IGLN.L failure mode: Yahoo's fast_info reports quote_type
    "EQUITY" for a gold ETC that has zero income-statement/balance-sheet
    data. A present-but-misleading quote_type must not short-circuit the
    statement-evidence check - only actual ebit/revenue/net_income/
    total_assets/total_liabilities count as evidence."""
    out = V.run_valuation({"quote_type": "EQUITY", "price": 47.32, "beta": 0.05})
    assert out["regime"]["regime"] == "unknown"
    assert out["models"] == {}


def test_regime_defaults_to_mature_profitable_only_with_explicit_quote_type():
    # explicit quote_type="EQUITY" plus real EBIT is still routed normally
    out = V.run_valuation(base_firm())
    assert out["regime"]["regime"] == "mature-profitable"


def test_no_silent_zero_on_missing_data():
    """Missing statements -> models exclude themselves with named reasons,
    fair_value stays None; nothing returns value == 0."""
    out = V.run_valuation({"quote_type": "EQUITY", "sector": "Technology",
                           "market_cap": 1000.0, "price": 10.0, "shares": 100.0})
    for m in out["models"].values():
        assert m["value"] != 0
        if not m["applicable"]:
            assert m["reason"]
    assert out["fair_value"] is None


def test_financial_sector_routes_to_equity_rim():
    f = base_firm(sector="Financial Services")
    out = V.run_valuation(f)
    assert out["regime"]["regime"] == "financial"
    assert "equity_rim" in out["models"] and "reoi" not in out["models"]


def test_fair_value_is_weighted_average_of_applicable():
    out = V.run_valuation(base_firm())
    pts = {n: m for n, m in out["models"].items()
           if m["applicable"] and m["value"] is not None and n in V.WEIGHTS}
    tw = sum(V.WEIGHTS[n] for n in pts)
    expected = sum(V.WEIGHTS[n] * m["value"] for n, m in pts.items()) / tw
    assert math.isclose(out["fair_value"], expected, rel_tol=1e-12)


def test_sensitivity_band_ordering():
    out = V.run_valuation(base_firm())
    s = out["sensitivity"]
    assert s is not None
    # lower rates -> higher value
    assert s["rate_minus_1pct"] > out["fair_value"] > s["rate_plus_1pct"]


# ---------------------------------------------------------------- flagged items


def test_core_ebit_check_flags_material_unusual_items():
    f = base_firm(ebit_unusual_items=30.0)  # 30/200 = 15% > 10% threshold
    got = V.core_ebit_check(f)
    assert got["applicable"]
    assert math.isclose(got["details"]["ebit_core"], 170.0, rel_tol=1e-9)
    assert got["details"]["flag_material_unusual_items"] is True


def test_core_ebit_check_na_without_line_item():
    got = V.core_ebit_check(base_firm())
    assert got["applicable"] is False


def test_buyback_value_check_below_fair_value_no_flag():
    # 500 spent buying back 20 shares -> implied price 25, below fair value 30
    f = base_firm(buybacks=500.0, shares_prev=120.0, shares=100.0)
    got = V.buyback_value_check(f, fair_value=30.0)
    assert got["applicable"]
    assert math.isclose(got["details"]["implied_avg_price"], 25.0, rel_tol=1e-9)
    assert got["details"]["flag_buybacks_above_fair_value"] is False


def test_buyback_value_check_above_fair_value_flags():
    # implied price 25 vs a much lower fair value of 15 -> destroyed value
    f = base_firm(buybacks=500.0, shares_prev=120.0, shares=100.0)
    got = V.buyback_value_check(f, fair_value=15.0)
    assert got["details"]["flag_buybacks_above_fair_value"] is True


def test_buyback_value_check_na_when_shares_grew():
    f = base_firm(buybacks=500.0, shares_prev=100.0, shares=110.0)
    got = V.buyback_value_check(f, fair_value=30.0)
    assert got["applicable"] is False


def test_cash_conversion_cycle_golden():
    f = base_firm(inventory=120.0, payables=90.0)
    got = V.cash_conversion_cycle(f)
    cogs = 1000.0 - 500.0
    dso = 80.0 / 1000.0 * 365
    dio = 120.0 / cogs * 365
    dpo = 90.0 / cogs * 365
    assert math.isclose(got["details"]["dso"], dso, rel_tol=1e-9)
    assert math.isclose(got["details"]["dio"], dio, rel_tol=1e-9)
    assert math.isclose(got["details"]["dpo"], dpo, rel_tol=1e-9)
    assert math.isclose(got["details"]["cash_conversion_cycle"], dso + dio - dpo, rel_tol=1e-9)


def test_cash_conversion_cycle_omits_dio_dpo_when_missing():
    got = V.cash_conversion_cycle(base_firm())  # no inventory/payables fetched
    assert got["applicable"]
    assert "dio" not in got["details"] and "dpo" not in got["details"]
    assert "cash_conversion_cycle" not in got["details"]


# ---------------------------------------------------------------- tier 2


def test_data_coverage_score_counts_present_fields():
    f = base_firm()
    got = V.data_coverage_score(f, ("revenue", "ebit", "made_up_field_xyz"))
    assert math.isclose(got["value"], 2 / 3, rel_tol=1e-9)
    assert got["details"]["missing"] == ["made_up_field_xyz"]


def test_analyst_target_triangulation_vs_price():
    f = base_firm(analyst_target_mean=36.0, analyst_target_count=12.0)
    got = V.analyst_target_triangulation(f)
    assert got["applicable"]
    assert math.isclose(got["details"]["target_vs_price_pct"], 20.0, rel_tol=1e-9)


def test_analyst_target_triangulation_na_without_target():
    got = V.analyst_target_triangulation(base_firm())
    assert got["applicable"] is False


def test_analyst_target_never_feeds_fair_value():
    """Tier 2 explicitly wants this as a side-by-side check, not a blend
    input: fair_value must be identical with/without an analyst target."""
    f_no_target = base_firm()
    f_with_target = base_firm(analyst_target_mean=999.0)
    assert V.run_valuation(f_no_target)["fair_value"] == V.run_valuation(f_with_target)["fair_value"]


# ---------------------------------------------------------------- tier 3


def test_monte_carlo_range_brackets_point_estimate_and_is_reproducible():
    f = base_firm()
    point = V.reoi_value(f)["value"]
    got = V.monte_carlo_range(f, n=200, seed=42)
    assert got["applicable"]
    d = got["details"]
    assert d["p10"] < d["p50"] < d["p90"]
    assert d["n"] == 200
    # same seed -> identical band (no hidden nondeterminism)
    again = V.monte_carlo_range(f, n=200, seed=42)
    assert again["details"] == d
    # point estimate should land inside a reasonably wide band most of the time
    assert d["p10"] < point * 1.5


def test_run_valuation_threads_live_rf_and_erp():
    """A live rf/erp pulled from data.fetch_macro() must actually change model
    outputs, not just get recorded and ignored."""
    f = base_firm()
    default_run = V.run_valuation(f)
    shifted_run = V.run_valuation(f, rf=0.06, erp=0.07, rf_source="test", erp_source="test")
    assert shifted_run["fair_value"] != default_run["fair_value"]
    assert shifted_run["rates_used"] == {"rf": 0.06, "erp": 0.07,
                                         "rf_source": "test", "erp_source": "test"}
    assert "roce_decomposition" in shifted_run
    assert "quick_ratios" in shifted_run

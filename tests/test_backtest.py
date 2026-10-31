"""Tier 2 backtest harness: statistics-only tests on synthetic records (no
network). run_backtest() itself fetches live yfinance data and is exercised
manually against real tickers, not in this offline suite."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest as B  # noqa: E402
import valuation as V  # noqa: E402


def rec(ticker="AAA", date="2023-01-01", price=100.0, price_after=110.0,
       fair_value=105.0, upside_pct=5.0, model_values=None, direction_hit=True):
    model_values = model_values if model_values is not None else {"reoi": 105.0}
    return B.BacktestRecord(
        ticker=ticker, statement_date=date, price_at_valuation=price,
        price_after=price_after,
        realized_return_pct=(price_after - price) / price * 100 if price_after else None,
        fair_value=fair_value, upside_pct=upside_pct,
        model_values=model_values, direction_hit=direction_hit)


def test_summarize_backtest_hit_rate():
    records = [rec(direction_hit=True), rec(direction_hit=True), rec(direction_hit=False)]
    got = B.summarize_backtest(records)
    assert got["n"] == 3
    assert math.isclose(got["hit_rate"], 2 / 3, rel_tol=1e-9)


def test_summarize_backtest_empty():
    got = B.summarize_backtest([])
    assert got["n"] == 0
    assert got["hit_rate"] is None


def test_calibrate_weights_from_backtest_favors_lower_error_model():
    # reoi is dead-on (0% error), owner_earnings is off by 50% -> reoi should
    # get a much larger candidate weight than the current 0.5/0.2 split implies.
    records = [
        rec(price_after=110.0, model_values={"reoi": 110.0, "owner_earnings": 55.0}),
        rec(price_after=200.0, model_values={"reoi": 200.0, "owner_earnings": 100.0}),
    ]
    got = B.calibrate_weights_from_backtest(records)
    assert got["n"] == 4
    assert math.isclose(got["mean_abs_error_pct"]["reoi"], 0.0, abs_tol=1e-9)
    assert got["mean_abs_error_pct"]["owner_earnings"] > 0
    assert got["weights"]["reoi"] > got["weights"]["owner_earnings"]
    assert math.isclose(sum(got["weights"].values()), 1.0, rel_tol=1e-9)
    # mechanism only: must not have mutated the production constant
    assert V.WEIGHTS == {"reoi": 0.5, "equity_rim": 0.5, "aeg": 0.3, "owner_earnings": 0.2}


def test_calibrate_weights_from_backtest_no_data():
    got = B.calibrate_weights_from_backtest([rec(price_after=None)])
    assert got["weights"] is None
    assert got["n"] == 0


def test_calibrate_fade_from_backtest_computes_persistence_ratio():
    records = [
        rec(ticker="AAA", date="2021-01-01", model_values={"reoi": 100.0}),
        rec(ticker="AAA", date="2023-01-01", model_values={"reoi": 80.0}),
    ]
    got = B.calibrate_fade_from_backtest(records, sectors={"AAA": "Technology"})
    assert math.isclose(got["sample_persistence_by_sector"]["Technology"], 0.8, rel_tol=1e-9)
    assert "pilot only" in got["note"]


def test_calibrate_fade_from_backtest_skips_single_snapshot_tickers():
    got = B.calibrate_fade_from_backtest([rec(ticker="AAA")], sectors={"AAA": "Technology"})
    assert got["sample_persistence_by_sector"] == {}

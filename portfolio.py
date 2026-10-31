"""Optional personal portfolio context (`portfolio.yaml`, hand-edited by the
user, gitignored - real holdings never get committed; see portfolio.yaml.
example for the template): current holdings + cash + target weights, fed to
the trader/risk seats so a BUY recommendation is sized with awareness of
what's already held - "add 4% of a diversified slice" means something
different at 0% existing exposure to this ticker than at 35%.

Entirely optional: no portfolio.yaml file means no portfolio context, not a
pipeline-ending error - this is context enrichment, not a required input.

Named simplification: no live FX conversion. A holding entered as `shares`
is live-priced and assumed to already be in the portfolio's base currency;
a cross-currency holding should be entered as a pre-computed `value` in that
base currency instead (see the example file). Building a real FX layer for
a single-user tool with a handful of holdings is not worth the added
surface - this is a sizing/concentration aid, not brokerage reconciliation.

Concentration is the Herfindahl-Hirschman Index (HHI) over %-weights (cash
counted as its own "position") - a standard single-number diversification
measure (0 = infinitely diversified, 10000 = 100% one position), rather than
a fragile "top-N%" cutoff whose meaning shifts with how many holdings exist.
"""

from __future__ import annotations

from pathlib import Path

import yaml

import data as D

PORTFOLIO_PATH = Path(__file__).resolve().parent / "portfolio.yaml"


def load_portfolio(path: Path = PORTFOLIO_PATH) -> dict | None:
    """None (not an error) when the file doesn't exist - this feature is
    entirely opt-in."""
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text()) or {}
    holdings_raw = raw.get("holdings") or {}
    if not isinstance(holdings_raw, dict):
        raise ValueError("portfolio.yaml: 'holdings' must map ticker -> {shares: N} or {value: N}")
    holdings = {}
    for ticker, spec in holdings_raw.items():
        if not isinstance(spec, dict) or not any(k in spec for k in ("shares", "value")):
            raise ValueError(f"portfolio.yaml: holding '{ticker}' needs a 'shares' or 'value' key")
        holdings[str(ticker)] = {"shares": float(spec["shares"]) if "shares" in spec else None,
                                 "value": float(spec["value"]) if "value" in spec else None}
    return {"currency": raw.get("currency"), "cash": float(raw.get("cash") or 0.0),
            "holdings": holdings, "targets": {str(k): float(v) for k, v in (raw.get("targets") or {}).items()}}


def _hhi(weights_pct: list[float]) -> float:
    return round(sum(w ** 2 for w in weights_pct), 1)


def snapshot(portfolio: dict, current_ticker: str | None = None,
            current_price: float | None = None) -> dict:
    """Live-priced snapshot: per-holding weight %, cash %, HHI concentration,
    and (if current_ticker is supplied) its existing weight before any new
    buy. current_price lets the caller reuse a price it already fetched for
    the ticker under analysis instead of a second network round-trip for
    the one ticker that matters most to the decision at hand."""
    values: dict[str, float] = {}
    unpriced: list[str] = []
    for ticker, spec in portfolio["holdings"].items():
        if spec["value"] is not None:
            values[ticker] = spec["value"]
            continue
        is_current = current_ticker is not None and ticker.upper() == current_ticker.upper()
        price = current_price if is_current else D.spot_price(ticker)
        if price is None:
            unpriced.append(ticker)
            continue
        values[ticker] = spec["shares"] * price
    total = sum(values.values()) + portfolio["cash"]
    if total <= 0:
        return {"applicable": False, "reason": "no priceable holdings and no cash"}
    weights = {t: round(v / total * 100, 2) for t, v in values.items()}
    cash_pct = round(portfolio["cash"] / total * 100, 2)
    existing_pct = next((w for t, w in weights.items()
                        if current_ticker and t.upper() == current_ticker.upper()), 0.0)
    return {"applicable": True, "total_value": round(total, 2), "currency": portfolio.get("currency"),
            "cash_pct": cash_pct, "holdings_pct": weights,
            "concentration_hhi": _hhi([*weights.values(), cash_pct]),
            "current_ticker_existing_pct": existing_pct if current_ticker else None,
            "targets_pct": portfolio.get("targets") or {}, "unpriced_holdings": unpriced}

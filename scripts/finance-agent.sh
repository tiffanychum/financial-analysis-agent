#!/usr/bin/env bash
# Detached launcher for the finance research pipeline.
#
# Usage: finance-agent.sh <TICKER> [HORIZON_MONTHS]
#        finance-agent.sh macro
#        finance-agent.sh macro-backtest <TICKER> [HORIZON_MONTHS]
#
# HORIZON_MONTHS is the holding period (1-60), default 12.
#
# IMPORTANT: this is run BY the main agent (via exec) in response to a WhatsApp
# "analyze <TICKER>" or "macro" command. It must return IMMEDIATELY: the
# pipeline runs detached (nohup) and sends its own WhatsApp summary when done
# via `openclaw message send` — safe from a separate process; only synchronous
# gateway calls from inside the agent's own turn deadlock the session.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo root, wherever it was cloned
PY="$DIR/.venv/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python3)"  # venv preferred, system python3 fallback
USAGE="usage: finance-agent.sh <TICKER> [HORIZON_MONTHS] | macro | macro-backtest <TICKER> [HORIZON_MONTHS]"
ARG="${1:-}"
[[ -n "$ARG" ]] || { echo "$USAGE" >&2; exit 2; }

mkdir -p "$DIR/reports"
ARG_UP=$(echo "$ARG" | tr '[:lower:]' '[:upper:]')

# Validates an optional horizon and fills HORIZON_ARGS for pipeline.py.
horizon_args() {
    local h="${1:-}"
    HORIZON_ARGS=()
    [[ -z "$h" ]] && return 0
    [[ "$h" =~ ^[0-9]{1,2}$ ]] && (( 10#$h >= 1 && 10#$h <= 60 )) || { echo "invalid horizon: $h (1-60 months)" >&2; exit 2; }
    HORIZON_ARGS=(--horizon-months "$((10#$h))")
}

# sanitize: tickers are alnum plus . - = ^ (e.g. BRK-B, EQQQ.L, GC=F, ^VIX)
check_ticker() {
    [[ "$1" =~ ^[A-Za-z0-9.=^\-]{1,10}$ ]] || { echo "invalid ticker: $1" >&2; exit 2; }
}

# "macro" is a keyword, not a ticker: standalone Fed/liquidity/inflation/
# growth/dollar regime scan (tech-ETF + gold tilt), not a per-instrument
# committee run - fewer model calls, different framing in the ack message.
if [[ "$ARG_UP" == "MACRO" ]]; then
    LOG="$DIR/reports/.last-run-MACRO.log"
    nohup "$PY" "$DIR/pipeline.py" "MACRO" >"$LOG" 2>&1 &
    echo "🌐 Macro regime scan started (5 model calls, ~1-2 min). I'll message you the tech ETF / gold tilt when it's done; full report lands in finance-agents/reports/."
    exit 0
fi

# "macro-backtest" replays the macro rules over ~20 years of history - no
# model calls, just FRED/Yahoo downloads.
if [[ "$ARG_UP" == "MACRO-BACKTEST" ]]; then
    TICKER="${2:-}"
    [[ -n "$TICKER" ]] || { echo "$USAGE" >&2; exit 2; }
    check_ticker "$TICKER"
    horizon_args "${3:-}"
    TICKER_UP=$(echo "$TICKER" | tr '[:lower:]' '[:upper:]')
    LOG="$DIR/reports/.last-run-MACRO-BACKTEST-${TICKER_UP}.log"
    nohup "$PY" "$DIR/pipeline.py" "MACRO-BACKTEST" "$TICKER" ${HORIZON_ARGS[@]+"${HORIZON_ARGS[@]}"} >"$LOG" 2>&1 &
    echo "📈 Macro backtest of ${TICKER_UP} started (no model calls, ~1 min). I'll message you how each macro regime played out historically; full report lands in finance-agents/reports/."
    exit 0
fi

check_ticker "$ARG"
horizon_args "${2:-}"
HORIZON_NOTE="${2:-12}-month holding period"

LOG="$DIR/reports/.last-run-${ARG_UP}.log"
nohup "$PY" "$DIR/pipeline.py" "$ARG" ${HORIZON_ARGS[@]+"${HORIZON_ARGS[@]}"} >"$LOG" 2>&1 &
echo "🔬 Analysis of ${ARG_UP} (${HORIZON_NOTE}) started (12-17 model calls, ~8-15 min). I'll message you the summary when it's done; full report lands in finance-agents/reports/."

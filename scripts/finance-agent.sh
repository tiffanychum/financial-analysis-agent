#!/usr/bin/env bash
# Detached launcher for the finance research pipeline.
#
# Usage: finance-agent.sh <TICKER> [HORIZON_MONTHS]
#        finance-agent.sh macro
#        finance-agent.sh track [TICKER]
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
# Apple Silicon: an x86_64 parent (e.g. an Intel Node gateway under Rosetta) makes the
# universal Python start as x86_64, which cannot load the arm64 numpy/pandas wheels
# ("you should not try to import numpy from its source directory"). Force arm64.
ARCH_PREFIX=()
if [[ "$(uname -s)" == Darwin && "$(sysctl -n hw.optional.arm64 2>/dev/null)" == 1 ]]; then
    ARCH_PREFIX=(arch -arm64)
fi
USAGE="usage: finance-agent.sh <TICKER> [HORIZON_MONTHS] | macro | track [TICKER] | macro-backtest <TICKER> [HORIZON_MONTHS]"
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
    nohup ${ARCH_PREFIX[@]+"${ARCH_PREFIX[@]}"} "$PY" "$DIR/pipeline.py" "MACRO" >"$LOG" 2>&1 &
    echo "🌐 Macro regime scan started (5 model calls, ~1-2 min). I'll message you the tech ETF / gold tilt when it's done; full report lands in reports/."
    exit 0
fi

# "track" scores every saved report against real prices (no model calls, ~10s):
# hit rate by horizon, "I was wrong" breaches, paper book vs monthly buying.
if [[ "$ARG_UP" == "TRACK" ]]; then
    TICKER="${2:-}"
    TICKER_ARG=()
    if [[ -n "$TICKER" ]]; then
        check_ticker "$TICKER"
        TICKER_ARG=("$(echo "$TICKER" | tr '[:lower:]' '[:upper:]')")
    fi
    LOG="$DIR/reports/.last-run-TRACK.log"
    nohup ${ARCH_PREFIX[@]+"${ARCH_PREFIX[@]}"} "$PY" "$DIR/pipeline.py" "TRACK" ${TICKER_ARG[@]+"${TICKER_ARG[@]}"} >"$LOG" 2>&1 &
    echo "📈 Track record scan started (no model calls, ~15s). I'll message you how past calls did vs real prices; full report lands in reports/."
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
    nohup ${ARCH_PREFIX[@]+"${ARCH_PREFIX[@]}"} "$PY" "$DIR/pipeline.py" "MACRO-BACKTEST" "$TICKER" ${HORIZON_ARGS[@]+"${HORIZON_ARGS[@]}"} >"$LOG" 2>&1 &
    echo "📈 Macro backtest of ${TICKER_UP} started (no model calls, ~1 min). I'll message you how each macro regime played out historically; full report lands in reports/."
    exit 0
fi

check_ticker "$ARG"
horizon_args "${2:-}"
HORIZON_NOTE="${2:-12}-month holding period"

LOG="$DIR/reports/.last-run-${ARG_UP}.log"
nohup ${ARCH_PREFIX[@]+"${ARCH_PREFIX[@]}"} "$PY" "$DIR/pipeline.py" "$ARG" ${HORIZON_ARGS[@]+"${HORIZON_ARGS[@]}"} >"$LOG" 2>&1 &
echo "🔬 Analysis of ${ARG_UP} (${HORIZON_NOTE}) started (~14 model calls, ~6-12 min). I'll message you the summary when it's done; full report lands in reports/."

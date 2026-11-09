<!-- Paste into your OpenClaw workspace AGENTS.md (the agent's rules).
     Replace /path/to/financial-analysis-agent with where you cloned the repo. -->

## Finance analysis command (on-demand only)

On `analyze <TICKER> [MONTHS]` (case-insensitive; ticker = 1-10 chars of letters/digits/./-/=/^,
e.g. AAPL, QQQ, BRK-B, GC=F; MONTHS = optional holding period 1-60, default 12), run exactly:

```
/path/to/financial-analysis-agent/scripts/finance-agent.sh <TICKER> [MONTHS]
```

Pass MONTHS only if the user gave one (e.g. `analyze QQQ 6` -> `QQQ 6`).
Append `debate` only if the user said it (e.g. `analyze AAPL debate` -> `AAPL debate`); it adds an optional bull/bear debate and is off otherwise.

The script detaches the research pipeline and returns one line IMMEDIATELY - reply with that
line verbatim and END the turn. Do NOT wait for the analysis (~8-15 min); the pipeline saves the
full report under `reports/` and, if configured, sends its own WhatsApp summary. Do not run it
twice for the same message. If the script errors, reply with its error line and stop - no
retries. Never call the `openclaw` CLI yourself for this feature (gateway calls from your own
turn deadlock the session).

On `macro` (no ticker - standalone Fed/liquidity/inflation/growth/dollar regime scan), run
exactly `/path/to/financial-analysis-agent/scripts/finance-agent.sh macro`. Same rules; ~1-2 min.

On `track` or `track <TICKER>` (how did past analyses do against real prices: hit rate, "I was
wrong" breaches, paper book vs monthly buying), run exactly
`/path/to/financial-analysis-agent/scripts/finance-agent.sh track [TICKER]`. Same rules; no model
calls, ~15s. It messages a plain-English summary and saves `reports/TRACK-<ts>.md`.

On `macro-backtest <TICKER> [MONTHS]` (does the macro tilt historically predict this ticker's
forward returns), run exactly
`/path/to/financial-analysis-agent/scripts/finance-agent.sh macro-backtest <TICKER> [MONTHS]`.
Same rules; no model calls, ~20s.

<!-- Optional: add a one-line cheat-sheet entry to TOOLS.md:
     `analyze <TICKER> [MONTHS]` | `macro` | `track [TICKER]` | `macro-backtest <TICKER> [MONTHS]` -> financial-analysis-agent/scripts/finance-agent.sh -->

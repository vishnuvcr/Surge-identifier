# Phase 6.2 design

Phase 6.2 is an independent research branch and does not cancel or modify the running Phase 6.1 workflow.

## Non-negotiable methodology
- Preserve the full Phase 6.1 information set and point-in-time rules.
- Preserve target definitions, stop-first conflict handling, costs, liquidity rules, walk-forward refitting and NO-TRADE behavior.
- Increase historical coverage substantially (configuration is 1500 sessions / 1000-session model window; engine must use all clean repository history available rather than truncating to recent 420/300 sessions).
- Do not weaken filters or reduce model complexity to make runtime shorter.

## Runtime strategy
- Horizontal sharding by disjoint out-of-sample calendar blocks: independent blocks train only on data strictly before their prediction start.
- Matrix jobs run blocks concurrently (max 5 in the first pass).
- Each block produces an auditable trade/equity ledger; aggregation happens only after all blocks finish.
- Common historical price/F&O/event inputs remain identical. A later pass can materialize a single immutable feature cache once and distribute it to shard jobs, avoiding repeated feature construction/download work without changing model semantics.
- Additional future optimization: parallelize independent expert fits inside each monthly refit using joblib/processes, while keeping deterministic random seeds and identical training rows.
- No GPU substitution is assumed; the bottlenecks are likely feature construction and repeated CPU tree/OOF fitting. If PyTorch is later used for any master, batched GPU training can be enabled without changing labels or splits.

## Acceptance criteria
6.2 is not accepted merely because it finishes faster. It must match a sequential reference implementation on overlapping dates for:
1. feature values and row counts,
2. training cutoffs,
3. candidate universe,
4. expert predictions/ranking within numerical tolerance,
5. trade entry/TP/SL ordering and costs,
6. equity curve and drawdown statistics.

Only after equivalence is demonstrated should the wider historical run be used for performance conclusions.

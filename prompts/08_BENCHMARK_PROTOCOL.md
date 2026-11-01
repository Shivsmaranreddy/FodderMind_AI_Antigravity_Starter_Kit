# Prompt 08 — Benchmarking

Implement a fair benchmark protocol and update `docs/experiments/benchmark_protocol.md`. Compare methods on the same scenario sets and evaluation conditions. Keep training and evaluation scenarios separate where relevant. Use multiple seeds, record configurations and versions, and report failures.

Metrics should include solution feasibility, cost under the simulator's assumptions, constraint violations, stockouts/waste if represented, runtime, and peak memory where feasible. Include a transparent baseline and deterministic optimizer. Do not cherry-pick seeds or report only best cases. Do not claim a method is superior unless results support that conclusion. Save machine-readable summaries under `outputs/benchmarks/` and document that results are simulated.

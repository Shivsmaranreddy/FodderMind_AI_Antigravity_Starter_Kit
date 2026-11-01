# Prompt 07 — Conditional lightweight RL experiment

Only proceed if the simulation represents a genuinely sequential decision problem. Start with tabular Q-learning if the state/action space allows; do not use deep RL by default.

Compare at minimum:
1. a simple fixed/rule-based policy;
2. repeated deterministic optimization where applicable;
3. the RL policy.

Use matched scenarios, multiple independent random seeds, bounded episodes, compact logs, separate evaluation scenarios, and fair metrics. Include cost, feasibility, stockouts/waste where modelled, constraint violations, runtime, and memory. Enforce hard constraints outside the reward. Report mean and variability/uncertainty, not just the best run. If RL does not outperform the baseline, say so. Do not claim real-world benefit from synthetic simulation alone.

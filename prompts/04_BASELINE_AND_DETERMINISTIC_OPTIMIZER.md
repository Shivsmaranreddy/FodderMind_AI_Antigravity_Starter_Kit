# Prompt 04 — Baseline and deterministic optimizer

Implement a transparent baseline and a deterministic constrained cost optimizer using the approved schema. Minimize cost subject to explicitly documented and source-supported constraints. Choose a suitable lightweight solver only after checking compatibility; explain trade-offs.

Requirements:
- independent feasibility validator after optimization;
- clear infeasibility diagnostics;
- unit and nutrient-basis consistency;
- no unsafe or incomplete plan returned as feasible;
- input validation and deterministic reproducibility;
- tests for feasible, infeasible, malformed, boundary, and missing-data cases;
- example inputs clearly labelled synthetic if invented for testing.

Do not add RL yet. Run tests and report actual results, including failures. Stop for approval.

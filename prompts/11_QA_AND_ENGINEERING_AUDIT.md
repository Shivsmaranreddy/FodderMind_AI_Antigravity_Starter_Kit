# Prompt 11 — QA and engineering audit

Review the whole project for import/dependency errors, API/database correctness, unit consistency, data provenance, unsafe or infeasible outputs, RL reward loopholes, data leakage, reproducibility, performance, security, and misleading UI claims.

Update `docs/validation/qa_plan.md`, `docs/validation/validation_report.md`, and `docs/validation/known_limitations.md`. Add unit, integration, and edge-case tests where justified. Run tests and report the exact commands, counts, failures, and any tests not run. Do not claim success without execution. Check the project on low-spec hardware or explain what could not be verified. Do not make large rewrites without approval.

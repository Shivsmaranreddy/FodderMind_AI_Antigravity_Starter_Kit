# Prompt 05 — Scientific and safety audit

Audit the data model and optimizer for unsupported claims, unit mismatches, inconsistent as-fed/dry-matter basis, missing-value handling, infeasible outputs, and unsafe reliance on reward penalties.

Update `docs/science/validation_rules.md`, `docs/science/model_scope_and_limits.md`, and the data source register. Create independent validation rules and tests. Nutritional values and requirements must have credible traceable sources or be explicitly labelled assumptions/placeholders. Do not predict growth, milk yield, health, or welfare unless a validated biological model and supporting evidence are present. Report unresolved scientific questions. Do not proceed until critical failures are resolved or clearly documented.

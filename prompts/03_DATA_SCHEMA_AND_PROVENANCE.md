# Prompt 03 — Data schema and provenance

Design data structures for feed ingredients, goat/group profiles, inventory, prices, constraints, scenarios, and experiment results. Update `docs/data/data_dictionary.md` and `docs/data/data_source_register.csv`.

For each field specify type, unit, basis (such as as-fed or dry matter), source, date, evidence status, and missing-data handling. Distinguish sourced/verified, user-entered, assumed, and synthetic values. Do not fabricate values or silently replace missing values with zero. Add validation tests for units, ranges, invalid inputs, and missing data. Do not encode unsourced nutritional requirements as facts. Stop and report.

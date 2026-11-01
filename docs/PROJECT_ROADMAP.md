# FodderMind AI — Project Roadmap

## Mission

Develop a lightweight prototype for goat fodder planning that evaluates whether optimization and simulation can support affordable feed decisions under changing prices, availability, budgets, and documented nutritional constraints.

## Product boundaries

### In scope for the first prototype
- Feed ingredient catalogue with units, nutrient basis, price, availability, source, date, and evidence status
- Goat/group profile fields with sourced requirement values
- Transparent baseline ration or planning method
- Deterministic constrained cost optimizer
- Independent feasibility/safety validation
- Reproducible simulation of prices, stock, budgets, and procurement/feeding decisions
- Fair benchmarks and sensitivity analysis
- Lightweight dashboard/reporting
- SFIC demo with explicit limitations

### Out of scope until justified
- Claims about real animal growth, milk production, disease, or welfare from unvalidated simulation
- Computer vision, continuous IoT, multi-agent DRL, large language model APIs, or complex deep learning as initial dependencies
- Production veterinary/nutrition advice or autonomous real-farm control
- Fabricated farm interviews, measurements, savings, or validation results

## Phase gates

### Phase 0 — Scope and acceptance criteria
**Deliverables:** `docs/requirements.md`, `docs/architecture/architecture.md`, `docs/assumptions.md`, `docs/risks_and_assumptions.md`, `docs/implementation_roadmap.md`

**Exit criteria:** Problem, users, inputs, outputs, units, constraints, exclusions, and acceptance criteria are explicit. Human approval required before major code generation.

### Phase 1 — Data model and provenance
**Deliverables:** `docs/data/data_dictionary.md`, `docs/data/data_source_register.csv`, `docs/science/validation_rules.md`

**Exit criteria:** Every important field has a type, unit, basis, source/provenance field, and missing-data behavior. Synthetic and verified data are separated. No unsourced nutritional numbers are presented as facts.

### Phase 2 — Baseline and deterministic optimizer
**Deliverables:** baseline method, optimizer module, independent validator, tests, example input/output

**Exit criteria:** Valid input produces a reproducible plan; invalid and infeasible input is handled clearly; constraints are checked independently of optimization reward; units and nutrient basis are tested.

### Phase 3 — Simulation model
**Deliverables:** `src/foddermind/simulation/`, simulation specification, scenario configurations, reproducibility logs

**Exit criteria:** State, actions, transitions, time step, uncertainty assumptions, and metrics are documented. Synthetic outputs are labelled. The simulation does not silently pretend to model biology.

**Design note:** Use time steps that match decisions (e.g. daily feeding or weekly procurement) unless there is a defensible reason for finer resolution. Generating one row every second does not make a simulation more realistic.

### Phase 4 — RL experiment (conditional)
**Deliverables:** `docs/experiments/rl_experiment_plan.md`, small tabular Q-learning baseline, reproducible results

**Entry condition:** There is a sequential decision problem that the deterministic optimizer cannot adequately solve as a one-shot/static problem.

**Exit criteria:** Compare a simple policy, repeated deterministic optimization, and RL on matched scenarios and independent random seeds. Report mean, variability/uncertainty, constraint violations, runtime, and memory. If RL does not improve results, record that and retain the simpler method.

### Phase 5 — Robustness and sensitivity
**Deliverables:** sensitivity analysis across price, availability, budget, and documented nutrient uncertainty ranges

**Exit criteria:** Assumed ranges are labelled as assumptions; outcomes and failures are reported, not hidden. No arbitrary parameter ranges are presented as measured farm conditions.

### Phase 6 — Dashboard and reporting
**Deliverables:** minimal working screens, API integration, exports, error states

**Exit criteria:** Every chart and control uses actual API or experiment data. No fake metrics, fake buttons, or hard-coded success claims. Core functionality works without paid services.

### Phase 7 — QA and reproducibility
**Deliverables:** test report, performance notes, dependency list, setup instructions, known limitations

**Exit criteria:** Tests run; failures are reported accurately; setup is reproducible on target hardware. No claim that tests passed unless they were actually executed.

### Phase 8 — SFIC presentation
**Deliverables:** `sfic/problem_statement.md`, `sfic/technical_summary.md`, `sfic/demo_script.md`, `sfic/limitations_and_validation.md`, `sfic/estimated_costs.md`

**Exit criteria:** Demo is reproducible; claims match evidence; synthetic data is clearly labelled; comparison methods and limitations are explained; no fabricated interviews or outcomes.

## Recommended order

1. Documentation and approval
2. Data schema and validation
3. Baseline and deterministic optimizer
4. Tests
5. Simulation
6. RL only if justified
7. Sensitivity and benchmarks
8. Dashboard
9. QA
10. SFIC presentation

## Definition of done for each phase

- [ ] Required files updated
- [ ] Assumptions and evidence status documented
- [ ] Tests written and actually run where applicable
- [ ] Actual results and failures recorded
- [ ] No unsupported claims
- [ ] Changed files listed
- [ ] Next three actions listed
- [ ] Human approval obtained before proceeding to the next major phase

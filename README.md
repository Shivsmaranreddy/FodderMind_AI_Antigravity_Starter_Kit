# FodderMind AI — Antigravity Starter Kit

**Project:** Simulation-assisted goat fodder planning and optimization  
**Purpose:** Build a lightweight, evidence-conscious prototype that helps explore lower-cost feed plans while respecting documented nutritional and operational constraints.

## Current engineering strategy

1. Define the problem, assumptions, units, and evidence standards.
2. Build a data-quality and provenance layer.
3. Implement a deterministic constrained optimizer and a transparent baseline.
4. Test correctness and infeasible cases.
5. Build a bounded, reproducible simulation for sequential uncertainty.
6. Add tabular Q-learning only if the problem genuinely requires sequential learning.
7. Compare methods fairly; consider deep RL or multi-agent DRL only if evidence justifies them.
8. Build a minimal dashboard and SFIC demo after the core is tested.

## Important scientific limits

- Simulation outputs are **synthetic/model-generated**, not observations from real goats or farms.
- Do not claim improved growth, milk yield, health, or welfare unless an appropriate validated biological model and credible evidence support the claim.
- Nutrient requirements and feed composition must have traceable sources, units, basis (e.g. as-fed or dry-matter basis), date, and evidence status.
- Missing values must not silently become zero.
- Hard safety/feasibility checks must sit outside the reward function. A reward penalty alone is not a safety guarantee.
- Report negative or inconclusive results honestly. RL may not outperform a deterministic optimizer.

## Hardware and cost constraints

Development target: Windows 10 64-bit, Intel Pentium G4400 (2 cores / 2 threads), 8 GB RAM, SSD.

- Prefer small datasets, CPU-compatible libraries, tabular methods, bounded experiments, and batched logging.
- Do not require a GPU, paid API, cloud account, or credit card for the baseline prototype.
- Use cloud compute only if a later experiment genuinely requires it.
- Avoid installing large frameworks before there is a justified use case.

## How to use this kit with Antigravity

1. Open this folder as the project workspace.
2. Read `docs/PROJECT_ROADMAP.md` and `prompts/00_WORKING_RULES.md`.
3. Send prompts in order, one phase at a time. Do not paste all prompts at once.
4. Review each phase's deliverables and tests before moving on.
5. Commit or back up a known-good version before major changes.
6. Keep this README and the roadmap updated as the implementation changes.

## Repository map

- `docs/` — requirements, architecture, science, data, experiments, validation
- `prompts/` — phased instructions for Antigravity
- `src/foddermind/` — application source code
- `data/` — raw, synthetic, and processed data kept separate
- `tests/` — unit, integration, and fixture tests
- `outputs/` — generated benchmark results, figures, and reports
- `sfic/` — presentation, demo script, and project submission materials

## First milestone

Do **not** start with deep RL or MADRL. The first milestone is a documented, tested deterministic optimizer that either returns a feasible feed plan or clearly explains why no feasible plan exists.

## Data and repository hygiene

- Do not put passwords, API keys, private farm records, or personal information in Git.
- Do not overwrite/delete user work without approval.
- Keep generated outputs reproducible and label assumptions.
- Avoid committing large raw datasets or generated simulation dumps; store concise summaries and configuration files where possible.

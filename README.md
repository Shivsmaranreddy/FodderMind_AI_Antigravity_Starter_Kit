# FodderMind AI — Simulation-First Decision Twin

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE_PLACEHOLDER.txt)
[![Tests: 14 Passed](https://img.shields.io/badge/tests-14%20passed-brightgreen.svg)]()

**FodderMind AI** is a simulation-first computational framework designed for a school-level scientific innovation competition (SFIC / ISEF standard). It evaluates whether dynamic optimization and reinforcement learning can optimize smallholder fodder production and livestock feeding decisions without continuous, expensive physical IoT sensing.

---

## 🔬 Core Modeling Foundations

1. **Dual Fodder Production System (Decision 1.B):**
   - **Hydroponic Rack:** 7-day modular tray cycle with dry-matter respiration loss ($8\% - 15\%$ loss over 7 days per FAO/ICAR-NIANP), fresh water multiplication ($\approx 6.0\times$), and misting/irrigation water accounting.
   - **Perennial Field Plot:** Cut-and-carry fodder grass (e.g. Hybrid Napier) modeled with sigmoidal regrowth dynamics and 30-day harvest intervals.

2. **Pen-Level Ruminant Energetics (Decision 2.B):**
   - Simulates a homogeneous pen of growing meat goats (e.g. 5–10 head).
   - Voluntary Dry Matter Intake capacity ($DMI_{cap}$) determined by metabolic weight and physical rumen fill.
   - Basal metabolic maintenance energy ($ME_m = 0.428 \cdot BW^{0.75}\text{ MJ/day}$) and crude protein ($CP_m$) based on NRC (2007) / ICAR (2013) standards.
   - **Liebig's Law of the Minimum:** Daily tissue gain is limited by the joint availability of surplus energy ($ME_{gain}$) and surplus amino nitrogen ($CP_{gain}$), capped at a biological ceiling of $\approx 200\text{ g/day/head}$.

3. **Storage & Aerobic Spoilage (FIFO):**
   - First-In-First-Out queue tracking feed age and moisture-dependent decay. High-moisture green fodder expires within 3 days without preservation.

4. **Transparent Epistemological Boundaries:**
   - Sourced literature parameters (NRC 2007, ICAR 2013, FAO Feedipedia) are strictly separated from user-configurable research placeholders and synthetic trajectories.

---

## 📁 Repository Map

```text
FodderMind_AI_Antigravity_Starter_Kit/
├── app.py                         # Streamlit interactive web dashboard
├── requirements.txt               # Lightweight, CPU-bounded dependencies
├── pytest.ini                     # Test configuration
│
├── data/                          # Data layer & provenance
│   ├── feed_library.csv           # Sourced feed library (DM%, CP%, ME, NDF%, costs, sources)
│   └── processed/
│       └── feed_library.csv
│
├── models/                        # Typed data contracts & schema validation
│   ├── __init__.py
│   └── schemas.py                 # FeedItem, GoatPenProfile, FarmResources, DailyStepRecord
│
├── simulation/                    # Mechanistic simulation engine
│   ├── __init__.py
│   ├── fodder.py                  # HydroponicRack (7-day) & FieldFodderPlot (Napier)
│   ├── inventory.py               # FeedInventory FIFO queue & aerobic spoilage
│   ├── animal.py                  # GoatPenModel (NRC 2007 bioenergetics)
│   └── farm_simulator.py          # Daily synchronous stepping & financial accounting
│
├── algorithms/                    # Decision & planning policies
│   ├── __init__.py
│   ├── rule_based.py              # Extension baseline heuristic (staggered planting, 50/35/15 diet)
│   └── ration_lp.py               # Deterministic Constrained Optimizer (SciPy linprog Highs)
│
├── reports/                       # Reporting & benchmarking
│   ├── __init__.py
│   └── generator.py               # Markdown audits, CSV export, and comparative tables
│
└── tests/                         # Unit and integration test suite
    ├── unit/
    │   ├── test_schemas.py
    │   ├── test_fodder_simulation.py
    │   ├── test_inventory_fifo.py
    │   ├── test_animal_energetics.py
    │   └── test_ration_lp.py
    └── integration/
        └── test_full_simulation.py
```

---

## 🚀 Setup & Execution Guide

### Hardware Requirements
- **OS:** Windows 10 / 11 64-bit (or Linux / macOS)
- **CPU:** Intel Pentium G4400 (2 cores / 2 threads) or better
- **RAM:** 8 GB
- **Sensors / Paid APIs:** Zero physical sensors or cloud APIs required.

### 1. Installation
Clone the repository and install the lightweight CPU requirements:
```bash
python -m pip install -r requirements.txt
```

### 2. Running Unit & Integration Tests
Run pytest across all modular test suites:
```bash
python -m pytest -v
```
All 14 tests covering schemas, animal bioenergetics, hydroponic tray mass balance, FIFO inventory decay, and SciPy LP optimization execute in under 35 seconds on a 2-core CPU.

### 3. Launching the Interactive Web Dashboard
Launch the Streamlit interface locally:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Dashboard Features

1. **Sourced Feed Library Catalog:** Inspect verified nutritional compositions (DM%, CP%, ME MJ/kg DM, NDF%) with traceable citations, or upload a custom CSV for instant schema validation.
2. **Interactive Farm Configuration:** Configure goat herd size, initial weights, hydroponic rack capacity, field fodder acreage, and water/cash constraints.
3. **Trajectory Simulation:** Run multi-day operational forecasts using either the **Rule-Based Extension Baseline** or the **Constrained LP Optimizer**.
4. **Physiological Safety Audit:** Live alerts for Subacute Ruminal Acidosis ($NDF < 25\%$) and negative energy balance ($ME < ME_m$).
5. **Head-to-Head Benchmarking:** Compare the economic margin over feed cost (MOFC), live weight gain, and feed spoilage across matched pseudo-random seeds.
6. **Data Export:** One-click download of daily simulation trajectory CSV and scientific Markdown audit logs.

---

## 🛣️ Phased Roadmap Next Steps

- [x] **Phase 0:** Master Architecture, units, schemas, and epistemic boundaries.
- [x] **Phase 1 & 2:** Verified CSV data catalogues, deterministic LP optimizer, and rule-based baseline.
- [x] **Phase 3:** Forward mechanistic dual-fodder simulator, pen-level energetics, and FIFO spoilage engine.
- [ ] **Phase 4:** Kalman Filter state estimator for sparse, noisy weekly weigh-tape observations.
- [ ] **Phase 5:** Gymnasium environment (`FodderMindEnv-v0`) and candidate PPO reinforcement learning agent.
- [ ] **Phase 6:** 50-seed Monte Carlo sensitivity stress-testing and domain randomization.
- [ ] **Phase 7:** SFIC submission package and competition slide deck artifacts.

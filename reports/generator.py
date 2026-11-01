"""
generator.py - Generates scientific summaries, benchmark tables, and CSV audit reports.
"""

from __future__ import annotations

import os
from typing import Dict, List
import pandas as pd
from models.schemas import SimulationResult


def generate_summary_markdown(result: SimulationResult) -> str:
    """
    Renders an audit-ready Markdown summary of the simulation run.
    """
    df = result.trajectory_df
    avg_adg = df["daily_adg_g_head"].mean() if "daily_adg_g_head" in df else 0.0
    total_water = df["water_used_liters"].sum() if "water_used_liters" in df else 0.0

    status_acidosis = "PASSED" if result.acidosis_risk_days == 0 else "WARNING - ACIDIC DIET"
    status_deficit = "PASSED" if result.energy_deficit_days == 0 else "WARNING - DEFICIT"

    md = f"""# FodderMind AI — Simulation Experiment Audit Report

**Status:** {result.execution_status}  
**Simulation Horizon:** {result.config.horizon_days} days (dt = {result.config.daily_step_dt} day)  
**Pen Profile:** {result.pen_profile.head_count} goats ({result.pen_profile.breed})  

---

## 1. Key Performance Indicators (KPIs)

| Metric | Measured Value | Unit |
| :--- | :--- | :--- |
| **Initial Pen Weight** | {result.initial_pen_weight_kg:.2f} | kg |
| **Final Pen Weight** | {result.final_pen_weight_kg:.2f} | kg |
| **Total Live Weight Gain** | {result.total_pen_weight_gain_kg:.2f} | kg |
| **Mean Daily Gain per Head (ADG)** | {avg_adg:.1f} | g / head / day |
| **Gross Margin Over Feed Cost (MOFC)** | {result.gross_margin_over_feed_cost:.2f} | Currency |
| **Total Feed Expenditure** | {result.total_feed_cost:.2f} | Currency |
| **Total Water Consumed** | {total_water:.1f} | Liters |
| **Feed Spoilage Loss** | {result.total_spoilage_loss_kg_dm:.2f} | kg Dry Matter |

---

## 2. Physiological Safety and Risk Audit

| Constraint Check | Violations (Days) | Safety Threshold | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Subacute Ruminal Acidosis** | {result.acidosis_risk_days} | Dietary NDF >= 25.0% | {status_acidosis} |
| **Negative Energy Balance** | {result.energy_deficit_days} | ME_intake >= ME_maintenance | {status_deficit} |

---

*Note: All biological outputs are synthetic and generated via transparent NRC/ICAR models for comparative research. Do not deploy without licensed veterinary review.*
"""
    return md


def export_trajectory_csv(result: SimulationResult, output_path: str) -> str:
    """Exports daily simulation trajectory to a standardized CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.trajectory_df.to_csv(output_path, index=False)
    return output_path


def compare_simulation_results(results_map: Dict[str, SimulationResult]) -> pd.DataFrame:
    """
    Produces a comparative benchmark DataFrame across multiple strategies.
    """
    rows = []
    for policy_name, res in results_map.items():
        df = res.trajectory_df
        rows.append({
            "Policy": policy_name,
            "Total Gain (kg)": round(res.total_pen_weight_gain_kg, 2),
            "Mean ADG (g/head/d)": round(df["daily_adg_g_head"].mean(), 1),
            "Total Feed Cost": round(res.total_feed_cost, 2),
            "MOFC": round(res.gross_margin_over_feed_cost, 2),
            "Spoilage Loss (kg DM)": round(res.total_spoilage_loss_kg_dm, 2),
            "Acidosis Days": res.acidosis_risk_days,
            "Energy Deficit Days": res.energy_deficit_days,
        })
    return pd.DataFrame(rows)

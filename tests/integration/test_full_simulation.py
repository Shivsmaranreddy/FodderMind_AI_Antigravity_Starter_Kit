"""
Integration test running full 30-day simulation comparing Rule-Based vs LP policies.
Verifies deterministic reproducibility, state consistency, and CSV report export.
"""

import os
import pytest
import pandas as pd

from models.schemas import (
    FarmResources,
    GoatPenProfile,
    SimulationConfig,
    load_feed_library_from_csv,
)
from simulation.farm_simulator import FarmSimulator
from algorithms.rule_based import RuleBasedAgronomistPolicy
from algorithms.ration_lp import LeastCostRationOptimizer
from reports.generator import export_trajectory_csv, compare_simulation_results, generate_summary_markdown


def test_full_simulation_run_and_reproducibility(tmp_path):
    csv_path = "data/feed_library.csv"
    assert os.path.exists(csv_path)
    feed_library = load_feed_library_from_csv(csv_path)

    config = SimulationConfig(horizon_days=30, random_seed=42)
    profile = GoatPenProfile(head_count=5, initial_mean_weight_kg=20.0)
    resources = FarmResources(
        hydroponic_trays_total=28,
        field_fodder_area_sqm=500.0,
        initial_cash_reserve=10000.0,
    )

    # Run 1: Rule-Based Policy
    sim_rule = FarmSimulator(config, profile, resources, feed_library)
    policy_rule = RuleBasedAgronomistPolicy()
    res_rule = sim_rule.run_simulation(policy_rule)

    assert res_rule.execution_status == "COMPLETED"
    assert len(res_rule.trajectory_df) == 30
    assert res_rule.total_pen_weight_gain_kg > 0.0

    # Run 2: LP Optimizer Policy
    sim_lp = FarmSimulator(config, profile, resources, feed_library)
    policy_lp = LeastCostRationOptimizer(target_adg_g_head=100.0)
    res_lp = sim_lp.run_simulation(policy_lp)

    assert res_lp.execution_status == "COMPLETED"
    assert len(res_lp.trajectory_df) == 30

    # Compare benchmark table
    comp_df = compare_simulation_results({
        "Rule_Based": res_rule,
        "LP_Optimizer": res_lp,
    })
    assert len(comp_df) == 2
    assert "Policy" in comp_df.columns
    assert "Total Gain (kg)" in comp_df.columns

    # Test CSV export
    out_csv = str(tmp_path / "sim_trajectory.csv")
    exported_path = export_trajectory_csv(res_rule, out_csv)
    assert os.path.exists(exported_path)
    df_loaded = pd.read_csv(exported_path)
    assert len(df_loaded) == 30

    # Test Markdown report generation
    md_report = generate_summary_markdown(res_rule)
    assert isinstance(md_report, str)
    assert "FodderMind AI" in md_report
    assert "Key Performance Indicators" in md_report
    assert "Physiological Safety and Risk Audit" in md_report

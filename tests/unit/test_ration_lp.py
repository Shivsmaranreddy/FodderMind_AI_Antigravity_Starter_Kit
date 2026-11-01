"""
Unit tests for SciPy linear programming constrained ration formulation.
"""

import pytest
from models.schemas import FeedItem
from algorithms.ration_lp import LeastCostRationOptimizer


@pytest.fixture
def feed_library():
    return {
        "Green": FeedItem(
            name="Green",
            category="green_fodder_hydroponic",
            dry_matter_pct=16.0,
            crude_protein_pct_dm=14.0,
            metabolizable_energy_mj_kg_dm=10.5,
            ndf_pct_dm=40.0,
            unit_cost_as_fed=2.5,
        ),
        "Hay": FeedItem(
            name="Hay",
            category="dry_roughage",
            dry_matter_pct=90.0,
            crude_protein_pct_dm=5.0,
            metabolizable_energy_mj_kg_dm=7.0,
            ndf_pct_dm=65.0,
            unit_cost_as_fed=6.0,
        ),
        "Grain": FeedItem(
            name="Grain",
            category="concentrate",
            dry_matter_pct=88.0,
            crude_protein_pct_dm=12.0,
            metabolizable_energy_mj_kg_dm=13.0,
            ndf_pct_dm=12.0,
            unit_cost_as_fed=22.0,
        ),
    }


def test_lp_feasible_ration(feed_library):
    optimizer = LeastCostRationOptimizer(target_adg_g_head=100.0)
    inventory = {"Green": 10.0, "Hay": 10.0, "Grain": 10.0}

    res = optimizer.solve_ration(
        dmi_capacity_kg_dm=3.5,
        me_m_req_mj=20.0,
        cp_m_req_g=200.0,
        head_count=5,
        available_inventory_dm=inventory,
        feed_library=feed_library,
    )

    assert res.is_feasible
    assert res.total_cost > 0.0
    assert sum(res.feed_allocation_kg_dm.values()) <= 3.5001
    assert res.projected_ndf_pct >= 24.99  # respects acidosis bound


def test_lp_infeasible_when_stock_empty(feed_library):
    optimizer = LeastCostRationOptimizer(target_adg_g_head=100.0)
    inventory = {"Green": 0.0, "Hay": 0.0, "Grain": 0.0}

    res = optimizer.solve_ration(
        dmi_capacity_kg_dm=3.5,
        me_m_req_mj=20.0,
        cp_m_req_g=200.0,
        head_count=5,
        available_inventory_dm=inventory,
        feed_library=feed_library,
    )

    assert not res.is_feasible

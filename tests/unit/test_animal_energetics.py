"""
Unit tests for goat pen energetics and growth equations (NRC 2007 standard).
"""

import pytest
from models.schemas import FeedItem, GoatPenProfile
from simulation.animal import GoatPenModel


@pytest.fixture
def test_pen_and_library():
    profile = GoatPenProfile(
        head_count=5,
        initial_mean_weight_kg=20.0,
        dmi_pct_bw_placeholder=3.5,
        me_maintenance_coeff=0.428,
        me_gain_per_kg=23.8,
        cp_maintenance_per_kg075=4.15,
    )
    pen = GoatPenModel(profile)

    library = {
        "Hay": FeedItem(
            name="Hay",
            category="dry_roughage",
            dry_matter_pct=90.0,
            crude_protein_pct_dm=6.0,
            metabolizable_energy_mj_kg_dm=7.5,
            ndf_pct_dm=65.0,
            unit_cost_as_fed=6.0,
        ),
        "Grain": FeedItem(
            name="Grain",
            category="concentrate",
            dry_matter_pct=88.0,
            crude_protein_pct_dm=18.0,
            metabolizable_energy_mj_kg_dm=12.5,
            ndf_pct_dm=15.0,
            unit_cost_as_fed=25.0,
        ),
    }
    return pen, library


def test_pen_maintenance_requirements(test_pen_and_library):
    pen, _ = test_pen_and_library
    # Total pen weight = 5 * 20 = 100 kg
    # Individual metabolic weight = 20^0.75 = 9.457 kg
    # Pen metabolic weight = 5 * 9.457 = 47.287 kg
    assert pen.total_pen_weight_kg == 100.0
    expected_me_m = 0.428 * 5 * (20.0 ** 0.75)
    assert pen.daily_maintenance_me_mj == pytest.approx(expected_me_m, rel=1e-3)
    expected_dmi_cap = 100.0 * 0.035
    assert pen.daily_dmi_capacity_kg_dm == pytest.approx(expected_dmi_cap, rel=1e-3)


def test_positive_growth_on_balanced_ration(test_pen_and_library):
    pen, library = test_pen_and_library
    initial_weight = pen.mean_weight_kg

    # Offer balanced ration satisfying ME and CP
    ration = {"Hay": 2.0, "Grain": 1.5}  # Total 3.5 kg DM (matches capacity)
    audit = pen.digest_and_grow(ration, library)

    assert audit.dmi_consumed_kg_dm == pytest.approx(3.5, rel=1e-3)
    assert audit.me_consumed_mj > audit.me_maintenance_required_mj
    assert audit.cp_consumed_g > audit.cp_maintenance_required_g
    assert audit.daily_gain_pen_kg > 0.0
    assert audit.daily_adg_head_g > 0.0
    assert pen.mean_weight_kg > initial_weight
    assert not audit.energy_deficit


def test_weight_loss_on_starvation(test_pen_and_library):
    pen, library = test_pen_and_library
    initial_weight = pen.mean_weight_kg

    # Offer zero feed
    audit = pen.digest_and_grow({}, library)
    assert audit.daily_gain_pen_kg < 0.0
    assert pen.mean_weight_kg < initial_weight
    assert audit.energy_deficit

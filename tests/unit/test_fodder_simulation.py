"""
Unit tests for fodder models (HydroponicRack and FieldFodderPlot).
"""

import pytest
from simulation.fodder import HydroponicRack, FieldFodderPlot


def test_hydroponic_rack_seeding_and_cycle():
    rack = HydroponicRack(total_trays=14, seed_kg_per_tray=1.5, cycle_days=7)
    assert rack.active_trays_count == 0
    assert rack.ready_trays_count == 0

    # Seed 4 trays
    seeded, seed_kg, seed_dm = rack.seed_trays(4)
    assert seeded == 4
    assert seed_kg == 6.0
    assert rack.active_trays_count == 4

    # Step 6 days - should not harvest yet
    for _ in range(6):
        fresh, dm, water = rack.step_day()
        assert fresh == 0.0
        assert dm == 0.0
        assert water > 0.0

    # On day 7, the 4 trays should harvest
    fresh, dm, water = rack.step_day()
    assert fresh > 0.0
    assert dm > 0.0
    # Mass conservation: dry matter harvested should reflect ~10% respiration loss
    # 4 trays * 1.5 kg seed * 0.90 DM = 5.4 kg seed DM; 5.4 * 0.90 = 4.86 kg DM
    expected_dm = 4 * (1.5 * 0.90) * 0.90
    assert dm == pytest.approx(expected_dm, rel=1e-2)
    # Rack should be empty now
    assert rack.active_trays_count == 0


def test_field_plot_regrowth_and_harvest():
    plot = FieldFodderPlot(
        area_sqm=500.0,
        cutting_interval_days=30,
        mature_yield_kg_fresh_sqm=2.5,
    )
    # Starts at mature state
    assert plot.days_since_cut == 30
    standing_fresh = plot.current_standing_biomass_fresh_kg
    assert standing_fresh == pytest.approx(500.0 * 2.5, rel=0.05)

    # Harvest today
    fresh, dm, water = plot.step_day(harvest_today=True)
    assert fresh > 1000.0
    assert dm > 200.0
    assert plot.days_since_cut == 0

    # Step next day without harvest - biomass should be low (fresh regrowth)
    fresh_next, dm_next, _ = plot.step_day(harvest_today=False)
    assert fresh_next == 0.0
    assert plot.days_since_cut == 1

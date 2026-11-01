"""
Unit tests for FIFO feed inventory and spoilage dynamics.
"""

import pytest
from simulation.inventory import FeedInventory


def test_inventory_fifo_withdrawal():
    inv = FeedInventory()
    inv.add_stock("Maize_Grain", "concentrate", 20.0)
    inv.add_stock("Maize_Grain", "concentrate", 30.0)

    assert inv.get_available_dm("Maize_Grain") == 50.0

    # Withdraw 25 kg
    withdrawn = inv.withdraw_stock("Maize_Grain", 25.0)
    assert withdrawn == 25.0
    assert inv.get_available_dm("Maize_Grain") == 25.0


def test_green_fodder_spoilage_expiration():
    inv = FeedInventory()
    # Green fodder default max shelf life is 3 days
    inv.add_stock("Hydro_Maize", "green_fodder_hydroponic", 15.0)

    # Step day 1 and 2: no spoilage
    lost_d1 = inv.step_day()
    assert lost_d1 == 0.0
    lost_d2 = inv.step_day()
    assert lost_d2 == 0.0
    lost_d3 = inv.step_day()
    assert lost_d3 == 0.0

    # Step day 4: batch age exceeds 3 days, should spoil
    lost_d4 = inv.step_day()
    assert lost_d4 == 15.0
    assert inv.get_available_dm("Hydro_Maize") == 0.0

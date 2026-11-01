import pytest
from models.scientific_common import Quantity, PLACEHOLDER
from models.feed_inventory import FeedComposition, FeedInventory, FeedPortion, Ration
from models.fodder_model import FodderModel,FodderParameters


def q(v,u):
    return Quantity(v,u,PLACEHOLDER)


def test_all_harvest_handling_loss_and_total_storage_loss():
    crop = FodderModel(FodderParameters(q(100,'kg DM'),q(0,'1/day'),q(0,'kg DM'),
        q(0,'1/day'),q(1,'fraction')),q(10,'kg DM'))
    event = crop.harvest(100)
    assert event.harvest_loss_kg_dm == 10
    assert event.usable_harvest_kg_dm == 0
    assert crop.biomass_kg_dm == 0
    inventory = FeedInventory()
    inventory.receive(FeedComposition('x'),q(10,'kg DM'),q(1e300,'1/day'),None)
    assert inventory.advance_to(1e300).decay_kg_dm == 10
    assert inventory.stock_kg_dm == 0
    assert inventory.balance['error_kg_dm'] == 0


def test_expiry_respects_age_on_arrival():
    inv = FeedInventory(start_day=10)
    inv.receive(FeedComposition('x'),q(10,'kg DM'),q(0,'1/day'),q(3,'day'),age_days=2)
    assert inv.advance_to(10.5).total_kg_dm == 0
    assert inv.advance_to(11).expired_kg_dm == 10


def test_no_infinite_derived_nutrients():
    ration = Ration((FeedPortion(FeedComposition('x',me_mj_per_kg_dm=q(45,'MJ/kg DM')),1e308,PLACEHOLDER),))
    with pytest.raises(ValueError):
        ration.nutrient('me')

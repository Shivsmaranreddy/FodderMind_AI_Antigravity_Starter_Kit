import random
import pytest
from models.fodder_model import FodderModel, FodderParameters
from models.feed_inventory import FeedInventory, FeedComposition
from models.goat_model import GoatModel, GoatProfile
from models.scientific_common import Quantity, PLACEHOLDER, Evidence, Provenance


def q(v,u):
    return Quantity(v,u,PLACEHOLDER)


def test_field_to_inventory_to_consumption_mass_balance():
    crop = FodderModel(FodderParameters(q(100,'kg DM'),q(.08,'1/day'),q(5,'kg DM'),
                       q(.01,'1/day'),q(.1,'fraction')),q(40,'kg DM'))
    inventory = FeedInventory()
    feed = FeedComposition('synthetic forage',q(.2,'fraction'),q(9,'MJ/kg DM'),q(.12,'fraction'))
    goat = GoatModel(GoatProfile(q(25,'kg BW'),180,'female','local unverified','indigenous','growing'))
    growth = field_loss = handling_loss = consumed_dm = 0.
    rng = random.Random(10)
    for day in range(30):
        inventory.advance_to(day)
        growth += crop.grow(1,rng.uniform(.3,1)).growth_kg_dm
        field_loss += crop.apply_standing_losses(1).standing_loss_kg_dm
        cut = crop.harvest(rng.uniform(0,4))
        handling_loss += cut.harvest_loss_kg_dm
        inventory.receive(feed,q(cut.usable_harvest_kg_dm,'kg DM'),q(.04,'1/day'),q(3,'day'))
        eaten = inventory.withdraw(.8)
        consumed_dm += eaten.kg_dm
        audit = goat.assess_consumed(eaten)
        assert audit['predicted_gain_g_head_day'] is None
        b = inventory.balance
        sinks = crop.biomass_kg_dm + field_loss + handling_loss + consumed_dm + b['stock_kg_dm'] + b['decay_kg_dm'] + b['expired_kg_dm']
        assert 40+growth == pytest.approx(sinks,abs=1e-10)


def test_observed_receipts_do_not_turn_simulation_into_observation():
    inv = FeedInventory()
    observed = Provenance(Evidence.OBSERVED,'farm scale, batch 7',measured_on='2026-11-01')
    inv.receive(FeedComposition('unanalysed'),Quantity(5,'kg DM',observed),q(.1,'1/day'),q(3,'day'))
    assert inv.storage_assumptions()[0][1].provenance.status == Evidence.UNCERTAIN
    inv.advance_to(1)
    assert inv.snapshot().portions[0].quantity_provenance.status == Evidence.SYNTHETIC
    assert inv.withdraw(1).portions[0].quantity_provenance.status == Evidence.SYNTHETIC

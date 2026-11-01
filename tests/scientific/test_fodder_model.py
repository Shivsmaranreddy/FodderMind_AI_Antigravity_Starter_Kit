import math
import random
import pytest
from models.scientific_common import Quantity, PLACEHOLDER, Evidence
from models.fodder_model import FodderParameters, FodderModel


def q(value, unit, **kwargs):
    return Quantity(value, unit, PLACEHOLDER, **kwargs)


def model(initial=20):
    return FodderModel(FodderParameters(q(100,'kg DM'), q(.08,'1/day'),
        q(5,'kg DM'), q(.02,'1/day'), q(.1,'fraction')), q(initial,'kg DM'))


def test_exact_growth_and_partition():
    m = model()
    event = m.grow(3, .5)
    assert event.after_kg_dm == pytest.approx(100/(1+4*math.exp(-.08*.5*3)))
    split = model()
    for _ in range(6):
        split.grow(.5, .5)
    assert split.biomass_kg_dm == pytest.approx(m.biomass_kg_dm)
    assert abs(event.balance_error_kg_dm) < 1e-12


def test_losses_and_harvest_mass_balance():
    m = model()
    loss = m.apply_standing_losses(2)
    assert loss.after_kg_dm == pytest.approx(20*math.exp(-.04))
    harvest = m.harvest(1e6)
    assert m.biomass_kg_dm == pytest.approx(5)
    assert harvest.usable_harvest_kg_dm+harvest.harvest_loss_kg_dm == pytest.approx(harvest.harvested_kg_dm)
    assert harvest.balance_error_kg_dm == pytest.approx(0)
    assert m.harvest(1).harvested_kg_dm == 0


def test_extinction_capacity_zero_time_and_stress():
    assert model(0).grow(100,1).after_kg_dm == 0
    assert model(100).grow(100,1).after_kg_dm == 100
    assert model().grow(0,1).after_kg_dm == 20
    assert model().grow(100,0).after_kg_dm == 20
    assert model().grow(1e300,1).after_kg_dm == 100
    m = model(5)
    m.apply_standing_losses(100)
    assert m.biomass_kg_dm < 5  # cutting protection must not create DM after mortality
    assert m.harvest(10).harvested_kg_dm == 0


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), True, '2'])
def test_reject_invalid_input_without_mutation(bad):
    m = model()
    for method, args in [(m.harvest, (bad,)), (m.grow, (bad,1)), (m.apply_standing_losses,(bad,))]:
        with pytest.raises(ValueError):
            method(*args)
        assert m.biomass_kg_dm == 20


def test_units_and_physical_configuration():
    with pytest.raises(ValueError):
        FodderModel(model().parameters, q(1000,'kg DM'))
    with pytest.raises(ValueError):
        FodderModel(model().parameters, q(20,'kg fresh'))
    with pytest.raises(ValueError):
        model().grow(1,1.01)
    with pytest.raises(ValueError):
        FodderParameters(q(5,'kg DM'),q(.1,'1/day'),q(6,'kg DM'),q(0,'1/day'),q(0,'fraction'))


def test_seeded_parameter_uncertainty():
    rate = q(.08,'1/day',lower=.04,upper=.12)
    a, b = random.Random(7), random.Random(7)
    xs = [rate.sample(a) for _ in range(10)]
    assert xs == [rate.sample(b) for _ in range(10)]
    assert all(.04 <= x.value <= .12 and x.provenance.status == Evidence.SYNTHETIC for x in xs)


def test_random_event_conservation():
    rng, m = random.Random(91), model()
    for _ in range(1000):
        before = m.biomass_kg_dm
        g = m.grow(rng.random()*3, rng.random())
        l = m.apply_standing_losses(rng.random()*3)
        h = m.harvest(rng.random()*20)
        assert m.biomass_kg_dm == pytest.approx(before+g.growth_kg_dm-l.standing_loss_kg_dm-h.harvested_kg_dm)
        assert 0 <= m.biomass_kg_dm <= 100

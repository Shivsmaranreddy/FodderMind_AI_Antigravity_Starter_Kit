import math
import random
import pytest
from models.scientific_common import Quantity, PLACEHOLDER
from models.feed_inventory import FeedComposition, FeedInventory, FeedPortion, Ration


def q(v,u):
    return Quantity(v,u,PLACEHOLDER)


def feed(name='test forage', cp=.12):
    return FeedComposition(name,q(.2,'fraction'),q(9,'MJ/kg DM'),q(cp,'fraction'),q(.6,'fraction'))


def receive(inv, dm, composition=None, rate=0, life=None, age=0):
    return inv.receive(composition or feed(),q(dm,'kg DM'),q(rate,'1/day'),
                       None if life is None else q(life,'day'),age)


def test_units_nutrients_and_unknowns():
    inv = FeedInventory()
    inv.receive(feed(),q(10,'kg as-fed'),q(0,'1/day'),None)
    ration = inv.withdraw(2)
    assert ration.kg_dm == 2
    assert ration.portions[0].kg_as_fed_equivalent == 10
    assert ration.nutrient('cp').total == pytest.approx(.24)
    assert ration.nutrient('me').total == 18
    assert ration.nutrient('calcium').total is None
    mixed = Ration(ration.portions+(FeedPortion(FeedComposition('unanalysed'),1,PLACEHOLDER),))
    assert mixed.nutrient('me').known_subtotal == 18
    assert mixed.nutrient('me').unknown_feed_kg_dm == 1
    assert mixed.nutrient('me').total is None
    assert Ration(()).nutrient('me').total == 0


def test_fifo_keeps_batch_analysis():
    inv = FeedInventory()
    receive(inv,2,feed(cp=.1))
    receive(inv,4,feed(cp=.2))
    ration = inv.withdraw(3)
    assert [p.kg_dm for p in ration.portions] == [2,1]
    assert ration.nutrient('cp').total == pytest.approx(.4)
    assert inv.stock_kg_dm == 3
    assert inv.withdraw(100).kg_dm == 3
    assert inv.withdraw(1).kg_dm == 0
    assert abs(inv.balance['error_kg_dm']) < 1e-12


def test_expiry_boundary_decay_partition_and_no_reviving_feed():
    a, b = FeedInventory(), FeedInventory()
    for inv in (a,b):
        receive(inv,10,rate=.1,life=3)
    for t in (1,2,3,4):
        a.advance_to(t)
    loss = b.advance_to(4)
    assert a.balance == pytest.approx(b.balance)
    assert loss.decay_kg_dm == pytest.approx(10*(1-math.exp(-.3)))
    assert loss.expired_kg_dm == pytest.approx(10*math.exp(-.3))
    assert b.withdraw(100).kg_dm == 0
    with pytest.raises(ValueError):
        receive(b,1,life=3,age=3)


def test_named_withdrawal_zero_and_clock():
    inv = FeedInventory(start_day=2)
    receive(inv,2,feed('a'))
    receive(inv,3,feed('b'))
    assert inv.withdraw(1,'b').kg_dm == 1
    assert inv.withdraw(1,'missing').kg_dm == 0
    assert inv.withdraw(0).kg_dm == 0
    assert inv.advance_to(2).total_kg_dm == 0
    with pytest.raises(ValueError):
        inv.advance_to(1)
    assert inv.stock_kg_dm == 4


@pytest.mark.parametrize('bad', [-1,float('nan'),float('inf'),True,'2'])
def test_invalid_requests_atomic(bad):
    inv = FeedInventory()
    receive(inv,2)
    with pytest.raises(ValueError):
        inv.withdraw(bad)
    assert inv.stock_kg_dm == 2
    with pytest.raises(ValueError):
        inv.advance_to(bad)
    assert inv.day == 0


def test_invalid_composition_and_units():
    for dm in (0,1.1):
        with pytest.raises(ValueError):
            FeedComposition('x',dm_fraction_as_fed=q(dm,'fraction'))
    with pytest.raises(ValueError):
        FeedComposition('x',cp_fraction_dm=q(12,'fraction'))
    with pytest.raises(ValueError):
        FeedComposition('x',me_mj_per_kg_dm=q(9,'MJ/kg fresh'))
    with pytest.raises(ValueError):
        FeedComposition('x').as_fed_to_dm(1)
    inv = FeedInventory()
    with pytest.raises(ValueError):
        inv.receive(feed(),q(1,'g DM'),q(0,'1/day'),None)
    assert inv.stock_kg_dm == 0


def test_seeded_sequence_mass_and_bounds():
    def run():
        inv, rng = FeedInventory(), random.Random(42)
        for day in range(100):
            inv.advance_to(day)
            receive(inv,rng.uniform(0,20),rate=.03,life=4)
            inv.withdraw(rng.uniform(0,25))
            assert inv.stock_kg_dm >= 0
            assert abs(inv.balance['error_kg_dm']) < 1e-9
        return inv.balance
    assert run() == run()

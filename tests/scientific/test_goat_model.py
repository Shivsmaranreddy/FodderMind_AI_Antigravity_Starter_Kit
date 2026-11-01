from dataclasses import replace
import pytest
from models.scientific_common import Quantity, PLACEHOLDER, Provenance, Evidence
from models.feed_inventory import FeedComposition, FeedPortion, Ration
from models.goat_model import GoatProfile, GoatModel, UnsupportedAnimalError


def q(v,u):
    return Quantity(v,u,PLACEHOLDER)


def profile(**kwargs):
    defaults = dict(weight=q(25,'kg BW'),age_days=180,sex='female',breed='local cross (unverified)',
                    biotype='indigenous',stage='growing',head_count=5)
    defaults.update(kwargs)
    return GoatProfile(**defaults)


def ration(cp=.1, me=9, dm=3):
    f = FeedComposition('test',cp_fraction_dm=q(cp,'fraction'),me_mj_per_kg_dm=q(me,'MJ/kg DM'))
    return Ration((FeedPortion(f,dm,PLACEHOLDER),))


def test_published_equations_and_units():
    r = GoatModel(profile()).requirements(100)
    assert r.maintenance_me_mj_head_day == pytest.approx(.489*25**.75)
    assert r.target_gain_me_mj_head_day == pytest.approx(1.98)
    assert r.maintenance_mp_kg_head_day == pytest.approx(3.07*25**.75/1000)
    assert r.target_gain_mp_kg_head_day == pytest.approx(.029)
    assert r.mp_kg_herd_day == pytest.approx(r.mp_kg_head_day*5)
    assert r.me_mj_herd_day == pytest.approx(r.me_mj_head_day*5)
    assert all(v.provenance.status == Evidence.SOURCED for _,v in r.coefficients)


def test_biotype_and_non_linear_weight_scaling():
    indigenous = GoatModel(profile()).requirements(100)
    dairy = GoatModel(profile(biotype='dairy')).requirements(100)
    meat = GoatModel(profile(biotype='meat_boer_cross')).requirements(100)
    assert dairy.maintenance_me_mj_head_day > indigenous.maintenance_me_mj_head_day
    assert meat.target_gain_mp_kg_head_day == pytest.approx(.0404)
    doubled = GoatModel(profile(weight=q(50,'kg BW'))).requirements(100)
    assert doubled.maintenance_me_mj_head_day/indigenous.maintenance_me_mj_head_day == pytest.approx(2**.75)


def test_adult_maintenance_requires_intake_and_no_browse():
    m = GoatModel(profile(stage='mature_maintenance',age_days=900))
    with pytest.raises(ValueError):
        m.requirements()
    r = m.requirements(intake_dm_kg_head_day=.8)
    assert r.mp_kg_head_day == pytest.approx((.0267*800+1.031*25**.75+.2*25**.6)/1000)
    with pytest.raises(UnsupportedAnimalError):
        m.requirements(100,intake_dm_kg_head_day=.8)


def test_cp_is_not_mp_and_quantity_is_not_growth():
    m = GoatModel(profile())
    for feed in (ration(),ration(cp=.3,dm=30)):
        result = m.assess_consumed(feed,target_gain_g_head_day=100)
        assert result['mp_supply_kg_herd_day'] is None
        assert result['mp_deficit_kg_herd_day'] is None
        assert result['predicted_gain_g_head_day'] is None
        assert result['nutritionally_complete'] is None
        assert m.profile.weight.value == 25
    missing_me = Ration((FeedPortion(FeedComposition('unknown'),2,PLACEHOLDER),))
    assert m.assess_consumed(missing_me)['me_deficit_mj_herd_day'] is None


def test_zero_intake_explicit_mp_and_adjustment():
    m = GoatModel(profile())
    zero = m.assess_consumed(Ration(()))
    assert zero['mp_supply_kg_herd_day'] == 0
    assert zero['me_deficit_mj_herd_day'] > 0
    with pytest.raises(ValueError):
        m.assess_consumed(Ration(()),mp_supply_herd_day=q(.1,'kg MP/day'))
    result = m.assess_consumed(ration(),mp_supply_herd_day=q(.2,'kg MP/day'))
    assert result['mp_supply_kg_herd_day'] == .2
    base = m.requirements(100)
    adjusted = m.requirements(100,maintenance_me_multiplier=q(1.2,'factor'))
    assert adjusted.maintenance_me_mj_head_day == pytest.approx(base.maintenance_me_mj_head_day*1.2)
    assert adjusted.target_gain_me_mj_head_day == base.target_gain_me_mj_head_day


@pytest.mark.parametrize('kwargs', [dict(pregnant=True),dict(lactating=True),dict(weaned=False),
    dict(appreciable_browse=True),dict(age_days=600),dict(stage='lactating'),
    dict(stage='mature_maintenance',age_days=900,biotype='meat_boer_cross')])
def test_unsupported_domain_fails_explicitly(kwargs):
    with pytest.raises(UnsupportedAnimalError):
        profile(**kwargs)


@pytest.mark.parametrize('kwargs', [dict(weight=q(0,'kg BW')),dict(weight=q(25,'kg')),
    dict(head_count=1.5),dict(head_count=True),dict(age_days=0),dict(body_condition_score=6)])
def test_invalid_profile(kwargs):
    with pytest.raises(ValueError):
        profile(**kwargs)


def test_input_validation_and_reproducibility():
    m = GoatModel(profile())
    assert m.requirements(100) == m.requirements(100)
    for bad in (-1,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):
            m.requirements(bad)
    with pytest.raises(ValueError):
        m.assess_consumed(ration(),mp_supply_herd_day=q(200,'g MP/day'))
    with pytest.raises(ValueError):
        Provenance(Evidence.OBSERVED,'farm scale')
    with pytest.raises(ValueError):
        Provenance(Evidence.OBSERVED,'farm scale',measured_on='bad date')

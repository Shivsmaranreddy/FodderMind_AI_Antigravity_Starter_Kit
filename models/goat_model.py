"""Goat nutrient REQUIREMENTS and supply audit, not a live-weight growth model.

Sourced equations: Luo et al. (2004), Small Ruminant Research 53, 191-219,
doi:10.1016/j.smallrumres.2004.04.001 (energy and mature protein); Luo et al.
(2004), 53, 309-326, doi:10.1016/j.smallrumres.2004.04.003 (growing MP).

W is individual live BW in kg, G is a USER TARGET in g/head/day, D is actual or
planned intake in kg DM/head/day. ME_m=a*W^.75 MJ/head/day; ME_g=b*G/1000.
Growing MP=(3.07*W^.75+c*G)/1000 kg MP/head/day. Mature non-browse MP =
(0.0267*1000*D + 1.031*W^.75 + 0.2*W^.6)/1000 kg MP/head/day.
Metabolizable protein (MP) is NOT crude protein (CP). No fixed CP-to-MP ratio.
For mixed herds evaluate individual profiles/subgroups rather than mean BW.
"""
from dataclasses import dataclass
import math
from .scientific_common import Quantity, Provenance, Evidence, number, integer
from .feed_inventory import Ration

ENERGY_REFERENCE = 'https://doi.org/10.1016/j.smallrumres.2004.04.001'
PROTEIN_REFERENCE = 'https://doi.org/10.1016/j.smallrumres.2004.04.003'
# Values are sourced population estimates, not local confidence intervals.
# Meat classification here requires >=50% Boer ancestry, matching the source.
GROWING_COEFFICIENTS = {
    'meat_boer_cross': (.489, 23.1, .404),
    'dairy': (.580, 23.1, .290),
    'indigenous': (.489, 19.8, .290),
}


class UnsupportedAnimalError(ValueError):
    """The animal/diet lies outside the implemented evidence domain."""


@dataclass(frozen=True)
class GoatProfile:
    weight: Quantity                 # kg BW per animal, not total pen mass
    age_days: int
    sex: str                         # female, intact_male or castrate
    breed: str                       # record actual breed/cross, not inferred
    biotype: str                      # indigenous, dairy, meat_boer_cross
    stage: str                        # growing or mature_maintenance
    head_count: int = 1               # homogeneous subgroup only
    weaned: bool = True
    pregnant: bool = False
    lactating: bool = False
    appreciable_browse: bool = False
    body_condition_score: float | None = None  # recorded 1-5 scale, no invented correction

    def __post_init__(self):
        self.weight.checked('kg BW', positive=True)
        integer(self.age_days, 'age_days')
        integer(self.head_count, 'head_count')
        if self.sex not in {'female','intact_male','castrate'}:
            raise ValueError('unsupported sex label')
        if not isinstance(self.breed,str) or not self.breed.strip():
            raise ValueError('record breed/cross explicitly, or use "unknown"')
        if self.biotype not in GROWING_COEFFICIENTS:
            raise UnsupportedAnimalError('classify biotype with a nutritionist; Angora not supported')
        for name in ('weaned','pregnant','lactating','appreciable_browse'):
            if type(getattr(self,name)) is not bool:
                raise ValueError(f'{name} must be bool')
        if self.body_condition_score is not None:
            number(self.body_condition_score, 'body_condition_score',minimum=1,maximum=5)
        if self.stage not in {'growing','mature_maintenance'}:
            raise UnsupportedAnimalError('only growing and mature_maintenance are implemented')
        if not self.weaned or self.pregnant or self.lactating:
            raise UnsupportedAnimalError('preweaning, pregnancy and lactation require separate reviewed equations')
        if self.appreciable_browse:
            raise UnsupportedAnimalError('browse/tannin effects require a reviewed digestibility/protein model')
        if self.stage == 'growing' and self.age_days > 548:
            raise UnsupportedAnimalError('growing MP source covers postweaning to about 18 months (548-day guard)')
        if self.stage == 'mature_maintenance' and self.biotype == 'meat_boer_cross':
            raise UnsupportedAnimalError('selected mature energy coefficient supports indigenous/dairy, not a mature Boer default')


@dataclass(frozen=True)
class Requirements:
    maintenance_me_mj_head_day: float
    target_gain_me_mj_head_day: float
    maintenance_mp_kg_head_day: float
    target_gain_mp_kg_head_day: float
    head_count: int
    target_gain_g_head_day: float
    coefficients: tuple[tuple[str, Quantity], ...]
    warnings: tuple[str, ...]
    estimated_growth_g_head_day: None = None
    output_status: str = 'uncertain_model_estimate'

    @property
    def me_mj_head_day(self):
        return self.maintenance_me_mj_head_day+self.target_gain_me_mj_head_day

    @property
    def mp_kg_head_day(self):
        return self.maintenance_mp_kg_head_day+self.target_gain_mp_kg_head_day

    @property
    def me_mj_herd_day(self):
        return self.me_mj_head_day*self.head_count

    @property
    def mp_kg_herd_day(self):
        return self.mp_kg_head_day*self.head_count


def sourced(value, unit, reference):
    return Quantity(value,unit,Provenance(Evidence.SOURCED,reference,
                    'Population equation; transfer to Telangana animals is uncertain.'))


class GoatModel:
    def __init__(self, profile: GoatProfile):
        self.profile = profile

    def requirements(self, target_gain_g_head_day=0., intake_dm_kg_head_day=None,
                     maintenance_me_multiplier: Quantity | None = None):
        """Compute daily needs for a stated target; do not update body weight.

        Multiplier combines externally reviewed maintenance activity/environment
        adjustments. Absence means 1.0 with an explicit unadjusted warning, not
        a calibrated estimate for grazing or Telangana summer heat.
        """
        p, w = self.profile, self.profile.weight.value
        gain = number(target_gain_g_head_day, 'target_gain_g_head_day')
        if intake_dm_kg_head_day is not None:
            intake = number(intake_dm_kg_head_day, 'intake_dm_kg_head_day')
        else:
            intake = None
        factor = 1. if maintenance_me_multiplier is None else maintenance_me_multiplier.checked('factor',positive=True)
        warnings = [
            'Population-derived requirements are uncertain for local breeds and management.',
            'No mineral, vitamin, drinking-water, intake-capacity or health adequacy conclusion.',
            'Age and sex delimit/describe the animal; no unsupported sex or body-condition coefficient is applied.',
            'Target gain is a planning input, not a growth forecast; weight is unchanged.',
        ]
        if maintenance_me_multiplier is None:
            warnings.append('Unadjusted maintenance energy: review grazing, activity and Telangana heat stress.')
        elif maintenance_me_multiplier.provenance.status in {Evidence.UNCERTAIN,Evidence.SYNTHETIC}:
            warnings.append('Maintenance multiplier is a placeholder/scenario assumption, not a validated correction.')
        coefficients = []
        if p.stage == 'growing':
            a,b,c = GROWING_COEFFICIENTS[p.biotype]
            me_m, me_g = a*w**.75*factor, b*gain/1000.
            mp_m, mp_g = 3.07*w**.75/1000., c*gain/1000.
            coefficients.extend([
                ('maintenance_me',sourced(a,'MJ/kg BW^0.75/day',ENERGY_REFERENCE)),
                ('gain_me',sourced(b,'MJ/kg target gain',ENERGY_REFERENCE)),
                ('maintenance_mp',sourced(3.07,'g MP/kg BW^0.75/day',PROTEIN_REFERENCE)),
                ('gain_mp',sourced(c,'g MP/g target gain',PROTEIN_REFERENCE))])
        else:
            if gain != 0:
                raise UnsupportedAnimalError('mature model supports maintenance only, not adult weight gain')
            if intake is None:
                raise ValueError('mature MP calculation requires explicit intake kg DM/head/day')
            me_m, me_g = .462*w**.75*factor, 0.
            mp_m, mp_g = (.0267*1000*intake+1.031*w**.75+.2*w**.6)/1000., 0.
            coefficients.extend([
                ('maintenance_me',sourced(.462,'MJ/kg BW^0.75/day',ENERGY_REFERENCE)),
                ('fecal_mp',sourced(.0267,'g MP/g DM intake',ENERGY_REFERENCE)),
                ('urinary_mp',sourced(1.031,'g MP/kg BW^0.75/day',ENERGY_REFERENCE)),
                ('scurf_mp',sourced(.2,'g MP/kg BW^0.6/day',ENERGY_REFERENCE))])
        if maintenance_me_multiplier is not None:
            coefficients.append(('maintenance_me_multiplier',maintenance_me_multiplier))
        for value in (me_m,me_g,mp_m,mp_g,(me_m+me_g)*p.head_count,(mp_m+mp_g)*p.head_count):
            number(value,'computed requirement')
        return Requirements(me_m,me_g,mp_m,mp_g,p.head_count,gain,tuple(coefficients),tuple(warnings))

    def assess_consumed(self, consumed: Ration, *, target_gain_g_head_day=0.,
                        mp_supply_herd_day: Quantity | None = None,
                        maintenance_me_multiplier: Quantity | None = None):
        """Audit ACTUALLY CONSUMED daily herd ration, not offered inventory.
        MP supply may only come from a separate documented diet/digestion estimate.
        Missing MP or ME remains unknown; energy sufficiency cannot imply growth.
        """
        if not isinstance(consumed,Ration):
            raise ValueError('consumed must be a Ration')
        req = self.requirements(target_gain_g_head_day,
                                consumed.kg_dm/self.profile.head_count,
                                maintenance_me_multiplier)
        me, cp = consumed.nutrient('me'), consumed.nutrient('cp')
        mp = None if mp_supply_herd_day is None else mp_supply_herd_day.checked('kg MP/day')
        if consumed.kg_dm == 0:
            if mp is not None and mp > 0:
                raise ValueError('positive dietary MP cannot be supplied by an empty ration')
            mp = 0.
        if mp is not None and mp > consumed.kg_dm:
            raise ValueError('MP supply exceeds total consumed DM')
        return {
            'requirements':req,
            'consumed_kg_dm_herd_day':consumed.kg_dm,
            'me_supply':me,
            'cp_supply':cp,
            'mp_supply_kg_herd_day':mp,
            'mp_supply_provenance':None if mp_supply_herd_day is None else mp_supply_herd_day.provenance,
            'me_deficit_mj_herd_day':None if me.total is None else max(0.,req.me_mj_herd_day-me.total),
            'mp_deficit_kg_herd_day':None if mp is None else max(0.,req.mp_kg_herd_day-mp),
            'predicted_gain_g_head_day':None,
            'growth_status':'not_predicted_requires_independent_validation',
            'nutritionally_complete':None,
        }

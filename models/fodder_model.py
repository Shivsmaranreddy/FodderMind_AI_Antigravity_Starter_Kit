"""Standing fodder dry matter: growth, field loss and harvest are separate events.

State B, capacity K and protected cutting residual R: kg DM for the entire plot.
Potential growth: B' = K*B / [B+(K-B)*exp(-r*s*dt)], where r is 1/day,
s is a supplied dimensionless stress factor [0,1], dt is days. Zero B stays zero.
Standing loss: L=B*(1-exp(-lambda*dt)). Harvest removal H<=max(B-R,0),
usable harvest U=H*(1-handling_loss). No inventory is implicitly created.
Every event reports B_after=B_before+growth-field_loss-harvest_removed.
All crop parameters require evidence; logistic dynamics are phenomenological,
not a Telangana-calibrated yield or irrigation response model.
"""
from dataclasses import dataclass
import math
from .scientific_common import Quantity, number


@dataclass(frozen=True)
class FodderParameters:
    capacity: Quantity               # kg DM / whole plot, NOT kg fresh or kg/ha
    growth_rate: Quantity            # 1/day
    cutting_residual: Quantity       # kg DM; protects cutting, not mortality
    standing_loss_rate: Quantity     # 1/day, continuous hazard
    harvest_loss_fraction: Quantity  # fraction of removed DM

    def __post_init__(self):
        self.capacity.checked('kg DM', positive=True)
        self.growth_rate.checked('1/day')
        self.cutting_residual.checked('kg DM')
        self.standing_loss_rate.checked('1/day')
        self.harvest_loss_fraction.checked('fraction', maximum=1)
        if self.cutting_residual.value > self.capacity.value:
            raise ValueError('cutting residual exceeds plot capacity')


@dataclass(frozen=True)
class BiomassEvent:
    before_kg_dm: float
    after_kg_dm: float
    growth_kg_dm: float = 0.
    standing_loss_kg_dm: float = 0.
    harvested_kg_dm: float = 0.
    usable_harvest_kg_dm: float = 0.
    harvest_loss_kg_dm: float = 0.
    evidence_status: str = 'synthetic'

    @property
    def balance_error_kg_dm(self):
        return self.after_kg_dm - (self.before_kg_dm + self.growth_kg_dm
                                   - self.standing_loss_kg_dm - self.harvested_kg_dm)


class FodderModel:
    def __init__(self, parameters: FodderParameters, initial_biomass: Quantity):
        self.parameters = parameters
        self.initial_biomass = initial_biomass
        self._biomass = initial_biomass.checked('kg DM', maximum=parameters.capacity.value)

    @property
    def biomass_kg_dm(self):
        return self._biomass

    @property
    def harvestable_kg_dm(self):
        return max(0., self._biomass-self.parameters.cutting_residual.value)

    def grow(self, days, stress_factor):
        dt = number(days, 'days')
        stress = number(stress_factor, 'stress_factor', maximum=1.)
        b, k = self._biomass, self.parameters.capacity.value
        if b == 0 or dt == 0 or stress == 0 or self.parameters.growth_rate.value == 0:
            return BiomassEvent(b, b)
        # Divide first to avoid overflow in K*B; clamp only floating point roundoff.
        fraction = b/k
        denominator = fraction+(1-fraction)*math.exp(-self.parameters.growth_rate.value*stress*dt)
        after = k if denominator == 0 else k*(fraction/denominator)
        after = min(k, max(b, after))
        self._biomass = after
        return BiomassEvent(b, after, growth_kg_dm=after-b)

    def apply_standing_losses(self, days):
        dt = number(days, 'days')
        b = self._biomass
        loss = b*(-math.expm1(-self.parameters.standing_loss_rate.value*dt))
        self._biomass = max(0., b-loss)
        return BiomassEvent(b, self._biomass, standing_loss_kg_dm=loss)

    def harvest(self, requested_kg_dm):
        """Bound removal by available DM; unmet harvest is not manufactured."""
        requested = number(requested_kg_dm, 'requested_kg_dm')
        b = self._biomass
        removed = min(requested, self.harvestable_kg_dm)
        loss = removed*self.parameters.harvest_loss_fraction.value
        self._biomass = max(0., b-removed)
        return BiomassEvent(b, self._biomass, harvested_kg_dm=removed,
                            usable_harvest_kg_dm=removed-loss, harvest_loss_kg_dm=loss)

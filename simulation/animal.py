"""
animal.py - Pen-level goat nutrition and growth energetics (Decision 2.B).

Implements NRC 2007 / ICAR 2013 energetic standards:
- Rumen fill and dry matter intake capacity (DMI).
- Basal metabolic maintenance energy (ME_m) and crude protein (CP_m).
- Liebig's Law of the Minimum for tissue synthesis (protein vs energy limit).
- Ruminal acidosis safety monitoring (NDF >= 25.0%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from models.schemas import FeedItem, GoatPenProfile


@dataclass
class DailyNutritionAudit:
    """Nutritional balance and physiological health metrics for the pen."""
    dmi_capacity_kg_dm: float
    dmi_consumed_kg_dm: float
    me_consumed_mj: float
    me_maintenance_required_mj: float
    cp_consumed_g: float
    cp_maintenance_required_g: float
    dietary_ndf_pct: float
    daily_gain_pen_kg: float
    daily_adg_head_g: float
    acidosis_risk: bool
    energy_deficit: bool


class GoatPenModel:
    """
    Simulates a homogeneous pen of growing meat goats.
    """
    def __init__(self, profile: GoatPenProfile):
        self.profile = profile
        self.head_count = profile.head_count
        self.mean_weight_kg = profile.initial_mean_weight_kg
        self.age_days = profile.initial_age_days

    @property
    def total_pen_weight_kg(self) -> float:
        return self.head_count * self.mean_weight_kg

    @property
    def metabolic_body_weight_pen(self) -> float:
        """Pen aggregate metabolic body weight (BW^0.75)."""
        return self.head_count * (self.mean_weight_kg ** 0.75)

    @property
    def daily_dmi_capacity_kg_dm(self) -> float:
        """Aggregate pen voluntary dry matter intake capacity (kg DM/day)."""
        dmi_fraction = self.profile.dmi_pct_bw_placeholder / 100.0
        return self.total_pen_weight_kg * dmi_fraction

    @property
    def daily_maintenance_me_mj(self) -> float:
        """Aggregate pen maintenance metabolizable energy requirement (MJ ME/day)."""
        return self.profile.me_maintenance_coeff * self.metabolic_body_weight_pen

    @property
    def daily_maintenance_cp_g(self) -> float:
        """Aggregate pen maintenance crude protein requirement (g CP/day)."""
        return self.profile.cp_maintenance_per_kg075 * self.metabolic_body_weight_pen

    def digest_and_grow(
        self,
        offered_feed_dm: Dict[str, float],
        feed_library: Dict[str, FeedItem],
    ) -> DailyNutritionAudit:
        """
        Simulates ingestion, nutrient partition, and growth over 1 day.
        
        Args:
            offered_feed_dm: Mapping of feed_name -> kg Dry Matter offered.
            feed_library: Catalog of feed ingredient compositions.
            
        Returns:
            DailyNutritionAudit with detailed physiological metrics.
        """
        total_offered_dm = sum(offered_feed_dm.values())
        capacity_dm = self.daily_dmi_capacity_kg_dm

        # If offered exceeds physical rumen fill capacity, intake is pro-rated
        scale = 1.0
        if total_offered_dm > capacity_dm and total_offered_dm > 0:
            scale = capacity_dm / total_offered_dm

        consumed_dm = {k: v * scale for k, v in offered_feed_dm.items()}
        total_consumed_dm = sum(consumed_dm.values())

        # Aggregate ingested nutrients
        total_me = 0.0
        total_cp_g = 0.0
        total_ndf_kg = 0.0

        for name, dm_kg in consumed_dm.items():
            if name in feed_library and dm_kg > 0:
                item = feed_library[name]
                total_me += dm_kg * item.metabolizable_energy_mj_kg_dm
                total_cp_g += dm_kg * (item.crude_protein_pct_dm * 10.0)  # 1% of 1 kg DM = 10 g
                total_ndf_kg += dm_kg * (item.ndf_pct_dm / 100.0)

        diet_ndf_pct = (total_ndf_kg / total_consumed_dm * 100.0) if total_consumed_dm > 0 else 0.0

        # Energy & Protein Partitioning
        req_me = self.daily_maintenance_me_mj
        req_cp_g = self.daily_maintenance_cp_g

        surplus_me = total_me - req_me
        surplus_cp_g = total_cp_g - req_cp_g

        if surplus_me >= 0:
            # Positive energy balance -> tissue synthesis
            # Tissue energetic cost ~23.8 MJ/kg; protein cost ~280 g CP/kg empty body gain (NRC 2007)
            adg_energy = surplus_me / self.profile.me_gain_per_kg
            adg_protein = max(0.0, surplus_cp_g / 280.0)
            
            # Liebig's minimum between protein and energy
            potential_pen_gain_kg = min(adg_energy, adg_protein)
            # Biological ceiling: max ~200 g/day per head for growing meat goats
            max_pen_gain_kg = self.head_count * 0.200
            actual_pen_gain_kg = min(potential_pen_gain_kg, max_pen_gain_kg)
        else:
            # Negative energy balance -> body reserve mobilization
            # Reserve mobilization efficiency ~0.80
            actual_pen_gain_kg = surplus_me / (self.profile.me_gain_per_kg / 0.80)

        # Update pen state
        gain_per_head_kg = actual_pen_gain_kg / self.head_count
        self.mean_weight_kg = max(5.0, self.mean_weight_kg + gain_per_head_kg)
        self.age_days += 1

        acidosis_risk = (diet_ndf_pct < self.profile.min_dietary_ndf_pct) and (total_consumed_dm > 0.1)
        energy_deficit = surplus_me < 0

        return DailyNutritionAudit(
            dmi_capacity_kg_dm=capacity_dm,
            dmi_consumed_kg_dm=total_consumed_dm,
            me_consumed_mj=total_me,
            me_maintenance_required_mj=req_me,
            cp_consumed_g=total_cp_g,
            cp_maintenance_required_g=req_cp_g,
            dietary_ndf_pct=diet_ndf_pct,
            daily_gain_pen_kg=actual_pen_gain_kg,
            daily_adg_head_g=gain_per_head_kg * 1000.0,
            acidosis_risk=acidosis_risk,
            energy_deficit=energy_deficit,
        )

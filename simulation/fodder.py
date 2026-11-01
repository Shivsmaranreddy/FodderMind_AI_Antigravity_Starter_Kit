"""
fodder.py - Agronomic fodder production models (Decision 1.B: Dual Fodder Architecture).

Models:
1. HydroponicRack: 7-day vertical tray rack (germination, respiration DM loss, fresh multiplication, water uptake).
2. FieldFodderPlot: Perennial cut-and-carry field plot (Napier grass regrowth curve, rotation cutting, water footprint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class TrayState:
    """Individual hydroponic tray state."""
    age_days: int = 0
    seed_dry_matter_kg: float = 1.35  # e.g., 1.5 kg seed @ 90% DM
    is_active: bool = False


class HydroponicRack:
    """
    Vertical hydroponic rack with modular trays operating on a 7-day harvest cycle.
    
    Agronomic Mechanisms:
    - Fresh Weight Expansion: Uptake of water expands fresh weight ~5.0x - 6.5x over 7 days.
    - Conservation of Mass & Respiration DM Loss: Respiration consumes starch, causing
      an 8% - 15% reduction in total dry matter over the cycle (FAO 2017).
    - Daily Water Requirement: Water needed for seed soaking, misting, and transpiration.
    """
    def __init__(
        self,
        total_trays: int = 28,
        seed_kg_per_tray: float = 1.5,
        seed_dm_pct: float = 90.0,
        cycle_days: int = 7,
        dm_respiration_loss_pct: float = 10.0,
        fresh_multiplication_factor: float = 6.0,
        water_liters_per_tray_day: float = 1.5,
    ):
        self.total_trays = total_trays
        self.seed_kg_per_tray = seed_kg_per_tray
        self.seed_dm_pct = seed_dm_pct
        self.cycle_days = cycle_days
        self.dm_loss_fraction = dm_respiration_loss_pct / 100.0
        self.fresh_mult = fresh_multiplication_factor
        self.water_per_tray_day = water_liters_per_tray_day

        # Pre-allocate trays
        self.trays: List[TrayState] = [TrayState() for _ in range(total_trays)]

    @property
    def active_trays_count(self) -> int:
        return sum(1 for t in self.trays if t.is_active)

    @property
    def ready_trays_count(self) -> int:
        return sum(1 for t in self.trays if t.is_active and t.age_days >= self.cycle_days)

    def seed_trays(self, count: int) -> Tuple[int, float, float]:
        """
        Seeds empty trays.
        Returns (seeded_count, seed_kg_used, seed_cost).
        """
        empty_indices = [i for i, t in enumerate(self.trays) if not t.is_active]
        to_seed = min(count, len(empty_indices))
        seed_used = to_seed * self.seed_kg_per_tray
        seed_dm = seed_used * (self.seed_dm_pct / 100.0)

        for i in range(to_seed):
            idx = empty_indices[i]
            self.trays[idx] = TrayState(
                age_days=0,
                seed_dry_matter_kg=self.seed_kg_per_tray * (self.seed_dm_pct / 100.0),
                is_active=True,
            )
        return to_seed, seed_used, seed_dm

    def step_day(self) -> Tuple[float, float, float]:
        """
        Advances all active trays by 1 day.
        Harvests all trays that reach or exceed cycle_days.
        
        Returns:
            (harvested_fresh_kg, harvested_dm_kg, water_consumed_liters)
        """
        water_used = self.active_trays_count * self.water_per_tray_day
        harvested_fresh = 0.0
        harvested_dm = 0.0

        for i, t in enumerate(self.trays):
            if t.is_active:
                t.age_days += 1
                if t.age_days >= self.cycle_days:
                    # Harvest tray
                    dm_out = t.seed_dry_matter_kg * (1.0 - self.dm_loss_fraction)
                    fresh_out = self.seed_kg_per_tray * self.fresh_mult
                    harvested_fresh += fresh_out
                    harvested_dm += dm_out
                    # Reset tray to inactive/empty
                    self.trays[i] = TrayState(age_days=0, seed_dry_matter_kg=0.0, is_active=False)

        return harvested_fresh, harvested_dm, water_used


class FieldFodderPlot:
    """
    Perennial cut-and-carry field fodder plot (e.g., Hybrid Napier / Cenchrus purpureus).
    
    Agronomic Mechanisms:
    - Regrowth Kinetics: Sigmoidal regrowth after harvest reaching peak yield at cutting interval (~30 days).
    - Evapotranspiration Water Demand: Daily irrigation required based on area.
    - Rotational Cutting: When harvested, yields accumulated biomass and resets regrowth timer.
    """
    def __init__(
        self,
        area_sqm: float = 500.0,
        cutting_interval_days: int = 30,
        mature_yield_kg_fresh_sqm: float = 2.5,
        dry_matter_pct: float = 20.0,
        daily_evap_mm: float = 3.0,  # 3 mm/day = 3 L/m^2/day
    ):
        self.area_sqm = area_sqm
        self.cutting_interval_days = cutting_interval_days
        self.mature_yield_fresh_sqm = mature_yield_kg_fresh_sqm
        self.dry_matter_pct = dry_matter_pct
        self.daily_evap_mm = daily_evap_mm
        self.days_since_cut = cutting_interval_days  # initially mature

    @property
    def current_relative_biomass_fraction(self) -> float:
        """Fraction of maximum biomass accumulated [0.0 to 1.0]."""
        # Sigmoidal growth curve
        tau = self.days_since_cut
        midpoint = self.cutting_interval_days / 2.0
        k = 0.20
        frac = 1.0 / (1.0 + np.exp(-k * (tau - midpoint)))
        return float(np.clip(frac, 0.05, 1.0))

    @property
    def current_standing_biomass_fresh_kg(self) -> float:
        return self.area_sqm * self.mature_yield_fresh_sqm * self.current_relative_biomass_fraction

    @property
    def current_standing_biomass_dm_kg(self) -> float:
        return self.current_standing_biomass_fresh_kg * (self.dry_matter_pct / 100.0)

    def step_day(self, harvest_today: bool = False) -> Tuple[float, float, float]:
        """
        Advances the field plot by 1 day.
        
        Returns:
            (harvested_fresh_kg, harvested_dm_kg, irrigation_water_liters)
        """
        # Water requirement: area * evapotranspiration depth (1 mm = 1 L/m^2)
        # Assuming minimal baseline irrigation (e.g., drip or furrow efficiency factor 0.70)
        water_used = self.area_sqm * (self.daily_evap_mm * 0.5)

        harvested_fresh = 0.0
        harvested_dm = 0.0

        if harvest_today and self.days_since_cut >= 15:  # allow harvest after min 15 days
            harvested_fresh = self.current_standing_biomass_fresh_kg
            harvested_dm = self.current_standing_biomass_dm_kg
            self.days_since_cut = 0
        else:
            self.days_since_cut += 1

        return harvested_fresh, harvested_dm, water_used

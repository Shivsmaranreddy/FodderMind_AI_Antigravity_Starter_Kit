"""
rule_based.py - Standard agronomic heuristic policy (Extension Baseline).

Implements standard field practice:
- Staggered hydroponic tray seeding (Total trays / 7 per day).
- Rotational field cutting when paddock reaches 30 days maturity.
- Fixed nutritional ratio: 50% Green DM, 35% Dry Roughage DM, 15% Concentrate DM.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
from models.schemas import DailyFarmAction
from simulation.farm_simulator import FarmSimulator


class RuleBasedAgronomistPolicy:
    """
    Fixed heuristic baseline policy representing traditional livestock extension rules.
    """
    def __init__(
        self,
        green_dm_ratio: float = 0.50,
        roughage_dm_ratio: float = 0.35,
        concentrate_dm_ratio: float = 0.15,
    ):
        self.green_ratio = green_dm_ratio
        self.roughage_ratio = roughage_dm_ratio
        self.concentrate_ratio = concentrate_dm_ratio

    def __call__(self, day: int, sim: FarmSimulator) -> DailyFarmAction:
        # 1. Hydroponic tray seeding: maintain continuous 7-day pipeline
        trays_per_day = max(1, sim.resources.hydroponic_trays_total // 7)

        # 2. Field harvest: cut whenever plot has reached 30 days regrowth
        harvest_field = sim.field.days_since_cut >= sim.resources.field_cutting_interval_days

        # 3. Formulate ration based on pen intake capacity
        dmi_target = sim.pen.daily_dmi_capacity_kg_dm

        alloc_dm: Dict[str, float] = {}

        # Green component (prefer hydroponic maize, fallback to napier)
        target_green_dm = dmi_target * self.green_ratio
        avail_hydro = sim.inventory.get_available_dm("Hydroponic_Maize_Green")
        avail_napier = sim.inventory.get_available_dm("Hybrid_Napier_Green")

        feed_hydro = min(target_green_dm, avail_hydro)
        rem_green = target_green_dm - feed_hydro
        feed_napier = min(rem_green, avail_napier)

        if feed_hydro > 0:
            alloc_dm["Hydroponic_Maize_Green"] = feed_hydro
        if feed_napier > 0:
            alloc_dm["Hybrid_Napier_Green"] = feed_napier

        # Dry Roughage component (Sorghum Stover)
        target_roughage_dm = dmi_target * self.roughage_ratio
        avail_rough = sim.inventory.get_available_dm("Sorghum_Stover_Hay")
        feed_rough = min(target_roughage_dm, avail_rough)
        if feed_rough > 0:
            alloc_dm["Sorghum_Stover_Hay"] = feed_rough

        # Concentrate component (Wheat Bran or Maize Grain)
        target_conc_dm = dmi_target * self.concentrate_ratio
        avail_wheat_bran = sim.inventory.get_available_dm("Wheat_Bran")
        avail_maize_grain = sim.inventory.get_available_dm("Maize_Grain_Cracked")

        feed_wheat = min(target_conc_dm, avail_wheat_bran)
        rem_conc = target_conc_dm - feed_wheat
        feed_maize = min(rem_conc, avail_maize_grain)

        if feed_wheat > 0:
            alloc_dm["Wheat_Bran"] = feed_wheat
        if feed_maize > 0:
            alloc_dm["Maize_Grain_Cracked"] = feed_maize

        return DailyFarmAction(
            trays_to_seed=trays_per_day,
            harvest_field_today=harvest_field,
            feed_allocation_kg_dm=alloc_dm,
        )


class GymRuleBasedPolicy:
    """
    Rule-based baseline policy compatible with the Gymnasium agent interface.
    Outputs action vector Box(5,) targeting extension ratio:
    - 50% Green DM (25% Hydroponic, 25% Napier)
    - 35% Dry Roughage
    - 15% Concentrate
    - Continuous tray seeding rate (1.0 = maximum capacity)
    """
    def __init__(self):
        # Desired softmax weights: [0.25, 0.25, 0.35, 0.15]
        # ln(w) - max(ln(w)) normalized to [-1, 1] range:
        # ln(0.25)=-1.386, ln(0.35)=-1.050, ln(0.15)=-1.897
        # Shifted relative to roughage: [-0.336, -0.336, 0.0, -0.847, 1.0]
        self._action = np.array([-0.336, -0.336, 0.0, -0.847, 1.0], dtype=np.float32)

    def predict(
        self,
        obs: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        return self._action.copy(), None

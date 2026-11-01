"""
environment.py - Reproducible Gymnasium Environment for FodderMind AI.

Separates models for:
1. Fodder Biomass: Dual-fodder dynamics (7-day hydroponic tray cycle + perennial Napier field plot).
2. Feed Inventory: FIFO storage queue with aerobic spoilage decay.
3. Water Footprint: Daily hydroponic transpiration/misting + field irrigation demand.
4. Farm Financials: Market feed purchases, seed costs, water tariffs, and live animal valuation.
5. Goat Bioenergetics: NRC 2007 / ICAR 2013 intake capacity, basal maintenance, Liebig gain.

Design Specifications:
- Timescale: Discrete daily process steps (dt = 1.0 day). Biological states do not update sub-second.
- Observation Space: Box(16,) normalized features.
- Action Space: Box(5,) continuous logits with safe Softmax simplex projection and independent fiber shield.
- Termination: Severe emaciation (BW < 0.70 * BW_0) or farm insolvency.
- Truncation: Horizon reached (e.g. 60 days).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from models.schemas import (
    FarmResources,
    FeedItem,
    GoatPenProfile,
    SimulationConfig,
    load_feed_library_from_csv,
)
from simulation.animal import GoatPenModel
from simulation.fodder import FieldFodderPlot, HydroponicRack
from simulation.inventory import FeedInventory


class FodderMindGymEnv(gym.Env):
    """
    Gymnasium-compliant multi-component agro-livestock decision environment.
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        pen_profile: Optional[GoatPenProfile] = None,
        resources: Optional[FarmResources] = None,
        feed_library: Optional[Dict[str, FeedItem]] = None,
        scenario_name: str = "baseline",
    ):
        super().__init__()

        self.config = config or SimulationConfig()
        self.pen_profile = pen_profile or GoatPenProfile()
        self.resources = resources or FarmResources()
        self.scenario_name = scenario_name

        if feed_library is None:
            csv_path = os.path.join("data", "feed_library.csv")
            if os.path.exists(csv_path):
                self.feed_library = load_feed_library_from_csv(csv_path)
            else:
                raise FileNotFoundError(f"Feed library CSV missing at: {csv_path}")
        else:
            self.feed_library = feed_library

        # Key feed names mapped to action indices
        self.action_feed_keys = [
            "Hydroponic_Maize_Green",
            "Hybrid_Napier_Green",
            "Sorghum_Stover_Hay",
            "Maize_Grain_Cracked",
        ]

        # Action Space: Box(5,) continuous in [-1.0, 1.0]
        # a[0..3]: unnormalized logits for ration allocation (Softmax-projected)
        # a[4]: hydroponic tray seeding rate in [-1.0, 1.0] -> [0, max_seedable_trays]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        # Observation Space: Box(16,) continuous in [-1.0, 1.0]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

        # Internal state components
        self.hydroponics: Optional[HydroponicRack] = None
        self.field: Optional[FieldFodderPlot] = None
        self.inventory: Optional[FeedInventory] = None
        self.pen: Optional[GoatPenModel] = None

        self.current_day: int = 0
        self.cash: float = 0.0
        self.initial_pen_weight: float = 0.0
        self.consecutive_deficit_days: int = 0
        self.total_acidosis_days: int = 0
        self.total_water_consumed: float = 0.0
        self.total_feed_expenditure: float = 0.0
        self.total_spoilage_lost_dm: float = 0.0
        self.last_weather_temp: float = 25.0
        self.last_spot_price_grain: float = 24.0

        # Scenario variability parameters (Uncertain / Stochastic priors)
        self._rng = np.random.default_rng(self.config.random_seed)

    def _get_obs(self) -> np.ndarray:
        """Constructs normalized 16-dimensional observation vector."""
        bw = self.pen.mean_weight_kg
        norm_bw = np.clip((bw - 10.0) / 40.0 * 2.0 - 1.0, -1.0, 1.0)
        norm_dmi = np.clip(self.pen.daily_dmi_capacity_kg_dm / 10.0 * 2.0 - 1.0, -1.0, 1.0)

        green_dm, rough_dm, conc_dm = self.inventory.get_category_totals()
        norm_green = np.clip(green_dm / 50.0 * 2.0 - 1.0, -1.0, 1.0)
        norm_rough = np.clip(rough_dm / 100.0 * 2.0 - 1.0, -1.0, 1.0)
        norm_conc = np.clip(conc_dm / 100.0 * 2.0 - 1.0, -1.0, 1.0)

        active_trays = self.hydroponics.active_trays_count / max(1, self.resources.hydroponic_trays_total)
        ready_trays = self.hydroponics.ready_trays_count / max(1, self.resources.hydroponic_trays_total)
        field_progress = min(1.0, self.field.days_since_cut / max(1, self.resources.field_cutting_interval_days))

        norm_temp = np.clip((self.last_weather_temp - 25.0) / 15.0, -1.0, 1.0)
        norm_grain_price = np.clip((self.last_spot_price_grain - 24.0) / 12.0, -1.0, 1.0)
        norm_cash = np.clip(np.tanh(self.cash / 10000.0), -1.0, 1.0)
        norm_deficit = np.clip(self.consecutive_deficit_days / 10.0 * 2.0 - 1.0, -1.0, 1.0)

        day_angle = 2.0 * np.pi * (self.current_day % 365) / 365.0
        sin_day = np.sin(day_angle)
        cos_day = np.cos(day_angle)

        water_reserve_ratio = np.clip(self.total_water_consumed / max(1.0, self.resources.daily_water_limit_l * 30.0), 0.0, 1.0) * 2.0 - 1.0
        acidosis_ratio = np.clip(self.total_acidosis_days / max(1, self.config.horizon_days) * 2.0 - 1.0, -1.0, 1.0)

        obs = np.array([
            norm_bw,
            norm_dmi,
            norm_green,
            norm_rough,
            norm_conc,
            active_trays * 2.0 - 1.0,
            ready_trays * 2.0 - 1.0,
            field_progress * 2.0 - 1.0,
            norm_temp,
            norm_grain_price,
            norm_cash,
            norm_deficit,
            sin_day,
            cos_day,
            water_reserve_ratio,
            acidosis_ratio,
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Initialize sub-models
        self.hydroponics = HydroponicRack(
            total_trays=self.resources.hydroponic_trays_total,
            seed_kg_per_tray=self.resources.hydroponic_seed_per_tray_kg,
        )
        self.field = FieldFodderPlot(
            area_sqm=self.resources.field_fodder_area_sqm,
            cutting_interval_days=self.resources.field_cutting_interval_days,
            mature_yield_kg_fresh_sqm=self.resources.field_yield_per_cut_kg_fresh_sqm,
        )
        self.inventory = FeedInventory()
        self.pen = GoatPenModel(self.pen_profile)

        # Initial baseline inventory
        for name, item in self.feed_library.items():
            if "dry" in item.category or "concentrate" in item.category:
                self.inventory.add_stock(name, item.category, 40.0)

        self.current_day = 0
        self.cash = self.resources.initial_cash_reserve
        self.initial_pen_weight = self.pen.total_pen_weight_kg
        self.consecutive_deficit_days = 0
        self.total_acidosis_days = 0
        self.total_water_consumed = 0.0
        self.total_feed_expenditure = 0.0
        self.total_spoilage_lost_dm = 0.0

        # Scenario weather/price initialization
        if self.scenario_name == "drought":
            self.last_weather_temp = 36.0 + float(self._rng.normal(0, 2.0))
            self.last_spot_price_grain = 32.0 + float(self._rng.normal(0, 1.5))
        elif self.scenario_name == "price_spike":
            self.last_weather_temp = 26.0 + float(self._rng.normal(0, 1.5))
            self.last_spot_price_grain = 42.0 + float(self._rng.normal(0, 2.0))
        else:
            self.last_weather_temp = 25.0 + float(self._rng.normal(0, 1.5))
            self.last_spot_price_grain = 24.0 + float(self._rng.normal(0, 1.0))

        obs = self._get_obs()
        info = {
            "day": 0,
            "pen_weight_kg": self.pen.total_pen_weight_kg,
            "cash_balance": self.cash,
            "scenario": self.scenario_name,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one discrete daily step in the simulation.
        
        Args:
            action: Box(5,) continuous action.
            
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0)

        # 1. Action Decoding: Continuous Softmax Projection for Ration
        logits = action[0:4]
        # Numerical stability shift
        exp_logits = np.exp(logits - np.max(logits))
        simplex_weights = exp_logits / np.sum(exp_logits)  # sums strictly to 1.0

        # Tray seeding decision: [-1, 1] -> [0, max_seedable_trays_today]
        max_daily_trays = max(1, self.resources.hydroponic_trays_total // 7)
        seed_fraction = (action[4] + 1.0) / 2.0  # [0.0, 1.0]
        trays_to_seed = int(np.round(seed_fraction * max_daily_trays))

        # Field cutting heuristic: cut if standing biomass >= 80% and green stock is low
        field_harvest_today = (
            self.field.days_since_cut >= self.resources.field_cutting_interval_days
            and self.inventory.get_category_totals()[0] < 15.0
        )

        # 2. Advance Fodder Production
        seeded_count, seed_kg, _ = self.hydroponics.seed_trays(trays_to_seed)
        seed_cost = seed_kg * self.resources.hydroponic_seed_cost_kg
        self.cash -= seed_cost

        hydro_fresh, hydro_dm, hydro_water = self.hydroponics.step_day()
        if hydro_dm > 0:
            self.inventory.add_stock("Hydroponic_Maize_Green", "green_fodder_hydroponic", hydro_dm)

        field_fresh, field_dm, field_water = self.field.step_day(field_harvest_today)
        if field_dm > 0:
            self.inventory.add_stock("Hybrid_Napier_Green", "green_fodder_field", field_dm)

        step_water = hydro_water + field_water
        water_cost = step_water * self.resources.water_cost_per_l
        self.cash -= water_cost
        self.total_water_consumed += step_water

        # 3. Independent Physiological Safety Shield
        # Ensure dietary NDF >= 25% to prevent subacute ruminal acidosis
        # If concentrate weight is too high, shield clamps it and redistributes to roughage
        target_dmi = self.pen.daily_dmi_capacity_kg_dm
        shield_intervened = False

        # Compute raw projected dietary NDF
        raw_ndf_sum = 0.0
        for i, key in enumerate(self.action_feed_keys):
            if key in self.feed_library:
                raw_ndf_sum += simplex_weights[i] * (self.feed_library[key].ndf_pct_dm / 100.0)

        safe_weights = simplex_weights.copy()
        if raw_ndf_sum < 0.25:
            shield_intervened = True
            # Cap concentrate (index 3) and reallocate to dry roughage (index 2)
            max_conc = 0.20
            if safe_weights[3] > max_conc:
                excess = safe_weights[3] - max_conc
                safe_weights[3] = max_conc
                safe_weights[2] += excess

        # 4. Withdraw Feed Allocations (FIFO)
        offered_dm_dict: Dict[str, float] = {}
        daily_feed_cost = seed_cost
        feed_shortage_dm = 0.0

        for i, key in enumerate(self.action_feed_keys):
            req_dm = safe_weights[i] * target_dmi
            withdrawn = self.inventory.withdraw_stock(key, req_dm)
            offered_dm_dict[key] = withdrawn
            shortage = req_dm - withdrawn
            if shortage > 0.01:
                feed_shortage_dm += shortage

            if key in self.feed_library and withdrawn > 0:
                cost_consumed = withdrawn * self.feed_library[key].cost_per_kg_dm
                daily_feed_cost += cost_consumed

        self.cash -= (daily_feed_cost - seed_cost)
        self.total_feed_expenditure += daily_feed_cost

        # 5. Animal Digestion & Growth
        audit = self.pen.digest_and_grow(offered_dm_dict, self.feed_library)

        # 6. Inventory Decay / Spoilage
        spoiled_dm = self.inventory.step_day()
        self.total_spoilage_lost_dm += spoiled_dm

        # 7. Update Safety Flags & Counters
        if audit.energy_deficit:
            self.consecutive_deficit_days += 1
        else:
            self.consecutive_deficit_days = max(0, self.consecutive_deficit_days - 1)

        if audit.acidosis_risk:
            self.total_acidosis_days += 1

        # 8. Reward Calculation
        # R = Live Weight Value Gain - Direct Feed Cost - Direct Water Cost - Spoilage Penalty - Deficit Penalty
        pen_gain_kg = audit.daily_gain_pen_kg
        weight_gain_value = pen_gain_kg * self.resources.goat_meat_price_per_kg_live

        spoilage_penalty = spoiled_dm * 10.0
        shortage_penalty = feed_shortage_dm * 15.0
        deficit_penalty = 50.0 if audit.energy_deficit else 0.0
        acidosis_penalty = 100.0 if audit.acidosis_risk else 0.0

        reward = float(
            weight_gain_value
            - daily_feed_cost
            - water_cost
            - spoilage_penalty
            - shortage_penalty
            - deficit_penalty
            - acidosis_penalty
        )

        self.current_day += 1

        # 9. Episode Termination & Truncation
        terminated = False
        truncated = False

        # Termination 1: Severe Emaciation (BW drops > 30% from initial)
        if self.pen.total_pen_weight_kg < 0.70 * self.initial_pen_weight:
            terminated = True
            reward -= 500.0  # terminal penalty for animal welfare failure

        # Termination 2: Severe Insolvency
        if self.cash < -15000.0:
            terminated = True
            reward -= 500.0  # bankruptcy terminal penalty

        # Truncation: Horizon completed
        if self.current_day >= self.config.horizon_days:
            truncated = True

        obs = self._get_obs()
        info = {
            "day": self.current_day,
            "pen_weight_kg": self.pen.total_pen_weight_kg,
            "mean_adg_g_head": audit.daily_adg_head_g,
            "daily_feed_cost": daily_feed_cost,
            "daily_water_liters": step_water,
            "feed_shortage_dm": feed_shortage_dm,
            "spoilage_dm": spoiled_dm,
            "cash_balance": self.cash,
            "acidosis_risk": bool(audit.acidosis_risk),
            "energy_deficit": bool(audit.energy_deficit),
            "shield_intervened": bool(shield_intervened),
            "consecutive_deficit_days": self.consecutive_deficit_days,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"Day {self.current_day:02d} | "
            f"Pen Wt: {self.pen.total_pen_weight_kg:.1f}kg | "
            f"Cash: {self.cash:.1f} | "
            f"Water: {self.total_water_consumed:.1f}L | "
            f"Spoilage: {self.total_spoilage_lost_dm:.1f}kg"
        )

"""
farm_simulator.py - Synchronous daily farm simulation engine.

Integrates hydroponics, field fodder, FIFO inventory, goat pen energetics,
resource constraints (water, cash), and physiological safety audits.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional
import pandas as pd
import numpy as np

from models.schemas import (
    DailyFarmAction,
    DailyStepRecord,
    FarmResources,
    FeedItem,
    GoatPenProfile,
    SimulationConfig,
    SimulationResult,
)
from simulation.animal import GoatPenModel
from simulation.fodder import FieldFodderPlot, HydroponicRack
from simulation.inventory import FeedInventory


class FarmSimulator:
    """
    Simulates daily multi-component farm operations over a planning horizon.
    """
    def __init__(
        self,
        config: SimulationConfig,
        pen_profile: GoatPenProfile,
        resources: FarmResources,
        feed_library: Dict[str, FeedItem],
        initial_inventory_dm: Optional[Dict[str, float]] = None,
    ):
        self.config = config
        self.pen_profile = pen_profile
        self.resources = resources
        self.feed_library = feed_library

        # Component models
        self.hydroponics = HydroponicRack(
            total_trays=resources.hydroponic_trays_total,
            seed_kg_per_tray=resources.hydroponic_seed_per_tray_kg,
        )
        self.field = FieldFodderPlot(
            area_sqm=resources.field_fodder_area_sqm,
            cutting_interval_days=resources.field_cutting_interval_days,
            mature_yield_kg_fresh_sqm=resources.field_yield_per_cut_kg_fresh_sqm,
        )
        self.inventory = FeedInventory()
        self.pen = GoatPenModel(pen_profile)

        # Financial & Resource State
        self.cash = resources.initial_cash_reserve
        self.current_day = 0
        self.history: List[DailyStepRecord] = []

        # Populate initial stock
        if initial_inventory_dm:
            for feed_name, kg_dm in initial_inventory_dm.items():
                if feed_name in feed_library:
                    item = feed_library[feed_name]
                    self.inventory.add_stock(feed_name, item.category, kg_dm)
        else:
            # Sane baseline starting inventory
            for name, item in feed_library.items():
                if "dry" in item.category or "concentrate" in item.category:
                    self.inventory.add_stock(name, item.category, 50.0)

    def step(self, action: DailyFarmAction) -> DailyStepRecord:
        """
        Executes one simulation step (1 day).
        """
        day = self.current_day

        # 1. Seeding Hydroponic Trays
        seeded_count, seed_kg_used, seed_dm = self.hydroponics.seed_trays(action.trays_to_seed)
        seed_cost = seed_kg_used * self.resources.hydroponic_seed_cost_kg
        self.cash -= seed_cost

        # 2. Advance Hydroponics & Harvest
        hydro_fresh_kg, hydro_dm_kg, hydro_water_l = self.hydroponics.step_day()
        if hydro_dm_kg > 0:
            # Store in green inventory
            self.inventory.add_stock(
                "Hydroponic_Maize_Green",
                category="green_fodder_hydroponic",
                dry_matter_kg=hydro_dm_kg,
            )

        # 3. Advance Field Plot & Harvest
        field_fresh_kg, field_dm_kg, field_water_l = self.field.step_day(action.harvest_field_today)
        if field_dm_kg > 0:
            self.inventory.add_stock(
                "Hybrid_Napier_Green",
                category="green_fodder_field",
                dry_matter_kg=field_dm_kg,
            )

        # 4. Water Accounting
        total_water_used = hydro_water_l + field_water_l
        water_cost = total_water_used * self.resources.water_cost_per_l
        self.cash -= water_cost

        # 5. Withdraw Feeds for Pen Diet (FIFO)
        actual_fed_dm: Dict[str, float] = {}
        daily_feed_expenditure = 0.0

        for feed_name, requested_dm in action.feed_allocation_kg_dm.items():
            if requested_dm > 0:
                withdrawn_dm = self.inventory.withdraw_stock(feed_name, requested_dm)
                if withdrawn_dm > 0:
                    actual_fed_dm[feed_name] = withdrawn_dm
                    if feed_name in self.feed_library:
                        item = self.feed_library[feed_name]
                        # Compute cost of consumed feed
                        cost_consumed = withdrawn_dm * item.cost_per_kg_dm
                        daily_feed_expenditure += cost_consumed

        # Deduct purchased concentrate/hay expense
        self.cash -= daily_feed_expenditure

        # 6. Animal Digestion & Growth
        audit = self.pen.digest_and_grow(actual_fed_dm, self.feed_library)

        # 7. Inventory Spoilage Check
        spoiled_dm = self.inventory.step_day()

        # 8. Record Telemetry
        stored_green, stored_rough, stored_conc = self.inventory.get_category_totals()

        record = DailyStepRecord(
            day=day,
            mean_body_weight_kg=self.pen.mean_weight_kg,
            pen_total_weight_kg=self.pen.total_pen_weight_kg,
            daily_adg_g_head=audit.daily_adg_head_g,
            dmi_capacity_kg_dm=audit.dmi_capacity_kg_dm,
            dmi_consumed_kg_dm=audit.dmi_consumed_kg_dm,
            me_consumed_mj=audit.me_consumed_mj,
            me_maintenance_required_mj=audit.me_maintenance_required_mj,
            cp_consumed_g=audit.cp_consumed_g,
            cp_required_g=audit.cp_maintenance_required_g,
            dietary_ndf_pct=audit.dietary_ndf_pct,
            daily_feed_cost=daily_feed_expenditure + seed_cost,
            daily_water_cost=water_cost,
            cash_balance=self.cash,
            water_used_liters=total_water_used,
            hydroponic_harvest_kg_fresh=hydro_fresh_kg,
            field_harvest_kg_fresh=field_fresh_kg,
            stored_green_kg_dm=stored_green,
            stored_roughage_kg_dm=stored_rough,
            stored_concentrate_kg_dm=stored_conc,
            spoilage_loss_kg_dm=spoiled_dm,
            safety_violation_acidosis=audit.acidosis_risk,
            safety_violation_energy_deficit=audit.energy_deficit,
        )

        self.history.append(record)
        self.current_day += 1
        return record

    def run_simulation(
        self,
        policy_fn: Callable[[int, FarmSimulator], DailyFarmAction],
    ) -> SimulationResult:
        """
        Runs the simulation across the full horizon defined in config.
        """
        initial_pen_weight = self.pen.total_pen_weight_kg

        for _ in range(self.config.horizon_days):
            action = policy_fn(self.current_day, self)
            self.step(action)

        df = pd.DataFrame([r.__dict__ for r in self.history])

        final_pen_weight = self.pen.total_pen_weight_kg
        weight_gain = final_pen_weight - initial_pen_weight
        meat_revenue_potential = weight_gain * self.resources.goat_meat_price_per_kg_live
        total_feed_costs = float(df["daily_feed_cost"].sum())
        total_water_costs = float(df["daily_water_cost"].sum())
        mofc = meat_revenue_potential - total_feed_costs

        return SimulationResult(
            config=self.config,
            pen_profile=self.pen_profile,
            resources=self.resources,
            trajectory_df=df,
            total_feed_cost=total_feed_costs,
            total_water_cost=total_water_costs,
            initial_pen_weight_kg=initial_pen_weight,
            final_pen_weight_kg=final_pen_weight,
            total_pen_weight_gain_kg=weight_gain,
            gross_margin_over_feed_cost=mofc,
            total_spoilage_loss_kg_dm=float(df["spoilage_loss_kg_dm"].sum()),
            acidosis_risk_days=int(df["safety_violation_acidosis"].sum()),
            energy_deficit_days=int(df["safety_violation_energy_deficit"].sum()),
            execution_status="COMPLETED",
        )

"""
ration_lp.py - Deterministic Constrained Optimizer (SciPy linprog Baseline).

Solves the daily least-cost ration formulation subject to:
1. Voluntary Dry Matter Intake capacity upper bound (Rumen fill limit).
2. Energy adequacy (Metabolizable Energy ME >= Maintenance + Target Gain).
3. Protein adequacy (Crude Protein CP >= Maintenance + Target Gain).
4. Fiber safety constraint (Dietary NDF >= 25.0% to prevent acidosis).
5. Storage inventory availability bounds (0 <= u_k <= Inventory_k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog

from models.schemas import DailyFarmAction, FeedItem
from simulation.farm_simulator import FarmSimulator


@dataclass
class LPOptimizationResult:
    """Result of linear programming ration solver."""
    is_feasible: bool
    status_message: str
    feed_allocation_kg_dm: Dict[str, float]
    total_cost: float
    projected_me_mj: float
    projected_cp_g: float
    projected_ndf_pct: float


class LeastCostRationOptimizer:
    """
    Formulates and solves the daily least-cost ration linear program.
    """
    def __init__(self, target_adg_g_head: float = 100.0):
        self.target_adg_g = target_adg_g_head

    def solve_ration(
        self,
        dmi_capacity_kg_dm: float,
        me_m_req_mj: float,
        cp_m_req_g: float,
        head_count: int,
        available_inventory_dm: Dict[str, float],
        feed_library: Dict[str, FeedItem],
    ) -> LPOptimizationResult:
        """
        Solves the LP for a single day.
        """
        candidate_feeds = [f for f in feed_library.keys() if available_inventory_dm.get(f, 0.0) > 0.05]
        if not candidate_feeds:
            return LPOptimizationResult(
                is_feasible=False,
                status_message="No feed ingredients available in inventory.",
                feed_allocation_kg_dm={},
                total_cost=0.0,
                projected_me_mj=0.0,
                projected_cp_g=0.0,
                projected_ndf_pct=0.0,
            )

        n_feeds = len(candidate_feeds)
        # Cost vector (Currency / kg DM)
        c = np.array([feed_library[name].cost_per_kg_dm for name in candidate_feeds])

        # Target growth requirements
        # Energy: 23.8 MJ / kg gain * target gain pen
        pen_target_gain_kg = head_count * (self.target_adg_g / 1000.0)
        req_me_total = me_m_req_mj + (pen_target_gain_kg * 23.8)
        # Protein: ~280 g CP / kg gain (NRC 2007)
        req_cp_total = cp_m_req_g + (pen_target_gain_kg * 280.0)

        # Inequality constraints: A_ub @ x <= b_ub
        # 1. Total DM <= DMI_capacity: sum(u_k) <= dmi_capacity
        # 2. Total DM >= 0.80 * DMI_capacity: -sum(u_k) <= -0.80 * dmi_capacity
        # 3. Energy: -sum(ME_k * u_k) <= -req_me_total
        # 4. Protein: -sum(CP_k_g * u_k) <= -req_cp_total
        # 5. Fiber safety: sum(u_k * (0.25 - NDF_frac_k)) <= 0  <=>  sum(u_k * NDF_frac_k) >= 0.25 * sum(u_k)
        A_ub = []
        b_ub = []

        # 1. DM upper bound
        A_ub.append([1.0] * n_feeds)
        b_ub.append(dmi_capacity_kg_dm)

        # 2. DM lower bound (satiety)
        A_ub.append([-1.0] * n_feeds)
        b_ub.append(-0.75 * dmi_capacity_kg_dm)

        # 3. ME requirement
        me_coeffs = [-feed_library[name].metabolizable_energy_mj_kg_dm for name in candidate_feeds]
        A_ub.append(me_coeffs)
        b_ub.append(-req_me_total)

        # 4. CP requirement (g CP per kg DM = % CP * 10)
        cp_coeffs = [-(feed_library[name].crude_protein_pct_dm * 10.0) for name in candidate_feeds]
        A_ub.append(cp_coeffs)
        b_ub.append(-req_cp_total)

        # 5. Acidosis safety: sum(u_k * (0.25 - NDF_k/100)) <= 0
        ndf_coeffs = [(0.25 - (feed_library[name].ndf_pct_dm / 100.0)) for name in candidate_feeds]
        A_ub.append(ndf_coeffs)
        b_ub.append(0.0)

        # Upper bounds from inventory: 0 <= u_k <= Inventory_k
        bounds = [(0.0, available_inventory_dm[name]) for name in candidate_feeds]

        res = linprog(
            c,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            bounds=bounds,
            method="highs",
        )

        if res.success:
            alloc = {candidate_feeds[i]: float(res.x[i]) for i in range(n_feeds) if res.x[i] > 1e-4}
            tot_dm = sum(alloc.values())
            tot_me = sum(alloc[name] * feed_library[name].metabolizable_energy_mj_kg_dm for name in alloc)
            tot_cp = sum(alloc[name] * feed_library[name].crude_protein_pct_dm * 10.0 for name in alloc)
            tot_ndf = sum(alloc[name] * (feed_library[name].ndf_pct_dm / 100.0) for name in alloc)
            ndf_pct = (tot_ndf / tot_dm * 100.0) if tot_dm > 0 else 0.0

            return LPOptimizationResult(
                is_feasible=True,
                status_message="Optimal feasible least-cost ration found.",
                feed_allocation_kg_dm=alloc,
                total_cost=float(res.fun),
                projected_me_mj=tot_me,
                projected_cp_g=tot_cp,
                projected_ndf_pct=ndf_pct,
            )
        else:
            return LPOptimizationResult(
                is_feasible=False,
                status_message=f"LP Infeasible: {res.message}",
                feed_allocation_kg_dm={},
                total_cost=0.0,
                projected_me_mj=0.0,
                projected_cp_g=0.0,
                projected_ndf_pct=0.0,
            )

    def __call__(self, day: int, sim: FarmSimulator) -> DailyFarmAction:
        """
        Generates DailyFarmAction using LP ration solver with heuristic tray/field management.
        Falls back to rule-based ration if LP is infeasible due to severe inventory stockout.
        """
        trays_per_day = max(1, sim.resources.hydroponic_trays_total // 7)
        harvest_field = sim.field.days_since_cut >= sim.resources.field_cutting_interval_days

        available_inv = {name: sim.inventory.get_available_dm(name) for name in sim.feed_library}

        lp_res = self.solve_ration(
            dmi_capacity_kg_dm=sim.pen.daily_dmi_capacity_kg_dm,
            me_m_req_mj=sim.pen.daily_maintenance_me_mj,
            cp_m_req_g=sim.pen.daily_maintenance_cp_g,
            head_count=sim.pen.head_count,
            available_inventory_dm=available_inv,
            feed_library=sim.feed_library,
        )

        if lp_res.is_feasible:
            alloc = lp_res.feed_allocation_kg_dm
        else:
            # Emergency fallback: feed whatever is available proportionally
            alloc = {}
            tot_avail = sum(available_inv.values())
            cap = sim.pen.daily_dmi_capacity_kg_dm
            if tot_avail > 0:
                scale = min(1.0, cap / tot_avail)
                for name, dm in available_inv.items():
                    if dm > 0.05:
                        alloc[name] = dm * scale

        return DailyFarmAction(
            trays_to_seed=trays_per_day,
            harvest_field_today=harvest_field,
            feed_allocation_kg_dm=alloc,
        )

"""
schemas.py - Data models, dimensional units, and validation rules for FodderMind AI.

Every field has explicit SI/nutritional units, defined boundaries, and provenance tags.
Configurable biological placeholders are explicitly labeled as such until validated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd


@dataclass(frozen=True)
class FeedItem:
    """
    Feed ingredient specification on a Dry Matter (DM) basis.
    
    Attributes:
        name: Unique string identifier for the feed ingredient.
        category: Botanical/feed category ('green_fodder_hydroponic', 'green_fodder_field',
                  'dry_roughage', 'concentrate').
        dry_matter_pct: % Dry Matter of as-fed material (units: %, range: 5.0 - 95.0).
        crude_protein_pct_dm: Crude Protein as % of dry matter (units: % DM, range: 2.0 - 55.0).
        metabolizable_energy_mj_kg_dm: Metabolizable Energy per kg DM (units: MJ ME / kg DM, range: 4.0 - 16.0).
        ndf_pct_dm: Neutral Detergent Fiber as % of dry matter (units: % DM, range: 10.0 - 85.0).
        unit_cost_as_fed: Market or production cost per kg as-fed material (units: Currency / kg as-fed).
        provenance_source: Traceable literature citation, database URL, or survey source.
        evidence_status: Epistemic status ('sourced', 'configured_placeholder', 'verified_lab').
        notes: Contextual agronomic or storage notes.
    """
    name: str
    category: str
    dry_matter_pct: float
    crude_protein_pct_dm: float
    metabolizable_energy_mj_kg_dm: float
    ndf_pct_dm: float
    unit_cost_as_fed: float
    provenance_source: str = "Unspecified"
    evidence_status: str = "configured_placeholder"
    notes: str = ""

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("FeedItem 'name' must be a non-empty string.")
        valid_cats = {
            "green_fodder_hydroponic",
            "green_fodder_field",
            "dry_roughage",
            "concentrate",
            "supplement",
        }
        if self.category not in valid_cats:
            raise ValueError(f"Invalid category '{self.category}'. Must be one of {valid_cats}")

        if not (5.0 <= self.dry_matter_pct <= 98.0):
            raise ValueError(f"dry_matter_pct={self.dry_matter_pct}% is out of physical range [5.0, 98.0].")
        if not (1.0 <= self.crude_protein_pct_dm <= 65.0):
            raise ValueError(f"crude_protein_pct_dm={self.crude_protein_pct_dm}% is out of physical range [1.0, 65.0].")
        if not (3.0 <= self.metabolizable_energy_mj_kg_dm <= 16.0):
            raise ValueError(f"metabolizable_energy={self.metabolizable_energy_mj_kg_dm} MJ/kg DM is out of range [3.0, 16.0].")
        if not (5.0 <= self.ndf_pct_dm <= 90.0):
            raise ValueError(f"ndf_pct_dm={self.ndf_pct_dm}% is out of physical range [5.0, 90.0].")
        if self.unit_cost_as_fed < 0.0:
            raise ValueError(f"unit_cost_as_fed={self.unit_cost_as_fed} cannot be negative.")

    @property
    def cost_per_kg_dm(self) -> float:
        """Derived cost per kg of dry matter (Currency / kg DM)."""
        return self.unit_cost_as_fed / (self.dry_matter_pct / 100.0)


@dataclass
class GoatPenProfile:
    """
    Pen-level cohort specifications for growing meat goats.
    
    Attributes:
        breed: Breed or cross designation.
        head_count: Number of goats in the pen (Decision 2.B, integer >= 1).
        initial_mean_weight_kg: Initial live weight per head (units: kg, range: 10.0 - 50.0).
        initial_age_days: Approximate initial animal age in days.
        target_adg_g_day: Target Average Daily Gain per head (units: g/day, range: 20.0 - 250.0).
        dmi_pct_bw_placeholder: Dry matter intake capacity as % of body weight (units: %, default placeholder 3.5%).
        me_maintenance_coeff: NRC 2007 placeholder for maintenance ME per kg metabolic weight (units: MJ ME / BW^0.75 / day).
        me_gain_per_kg: Placeholder metabolizable energy requirement per kg empty body gain (units: MJ ME / kg gain).
        cp_maintenance_per_kg075: Placeholder crude protein maintenance requirement (units: g CP / BW^0.75 / day).
        min_dietary_ndf_pct: Minimum dietary NDF% to prevent subacute ruminal acidosis (units: % DM, standard >= 25.0%).
        evidence_status: Epistemic status of parameters ('sourced_prior', 'configured_placeholder').
    """
    breed: str = "Sirohi_Meat_Cross"
    head_count: int = 5
    initial_mean_weight_kg: float = 20.0
    initial_age_days: int = 120
    target_adg_g_day: float = 100.0
    dmi_pct_bw_placeholder: float = 3.5
    me_maintenance_coeff: float = 0.428
    me_gain_per_kg: float = 23.8
    cp_maintenance_per_kg075: float = 4.15
    min_dietary_ndf_pct: float = 25.0
    evidence_status: str = "sourced_prior"

    def __post_init__(self):
        if self.head_count < 1:
            raise ValueError("head_count must be at least 1.")
        if not (8.0 <= self.initial_mean_weight_kg <= 80.0):
            raise ValueError(f"initial_mean_weight_kg={self.initial_mean_weight_kg} kg is out of physiological range [8.0, 80.0].")
        if not (1.5 <= self.dmi_pct_bw_placeholder <= 6.0):
            raise ValueError(f"dmi_pct_bw_placeholder={self.dmi_pct_bw_placeholder}% is out of physiological range [1.5, 6.0].")
        if not (0.20 <= self.me_maintenance_coeff <= 0.80):
            raise ValueError(f"me_maintenance_coeff={self.me_maintenance_coeff} is out of expected range [0.20, 0.80].")
        if self.min_dietary_ndf_pct < 15.0:
            raise ValueError("min_dietary_ndf_pct cannot be safely below 15.0% without severe acidosis risk.")


@dataclass
class FarmResources:
    """
    Physical farm infrastructure, resource boundaries, and market prices.
    
    Attributes:
        hydroponic_trays_total: Total available hydroponic trays in rack system (integer >= 0).
        hydroponic_seed_per_tray_kg: Dry seed loaded per tray (units: kg seed / tray).
        hydroponic_seed_cost_kg: Purchase cost of clean germinable seed (units: Currency / kg seed).
        field_fodder_area_sqm: Dedicated land area for perennial cut-and-carry fodder (units: m^2, Decision 1.B).
        field_cutting_interval_days: Rotation interval between plot cuts (units: days).
        field_yield_per_cut_kg_fresh_sqm: Biomass yield per cut (units: kg fresh / m^2).
        initial_cash_reserve: Liquid capital available at day 0 (units: Currency).
        daily_water_limit_l: Physical or quota constraint on daily farm water (units: L / day).
        water_cost_per_l: Marginal cost of fresh water (units: Currency / L).
        goat_meat_price_per_kg_live: Farmgate sale price per kg live weight (units: Currency / kg live).
    """
    hydroponic_trays_total: int = 28
    hydroponic_seed_per_tray_kg: float = 1.5
    hydroponic_seed_cost_kg: float = 22.0
    field_fodder_area_sqm: float = 500.0
    field_cutting_interval_days: int = 30
    field_yield_per_cut_kg_fresh_sqm: float = 2.5
    initial_cash_reserve: float = 15000.0
    daily_water_limit_l: float = 300.0
    water_cost_per_l: float = 0.03
    goat_meat_price_per_kg_live: float = 350.0

    def __post_init__(self):
        if self.hydroponic_trays_total < 0:
            raise ValueError("hydroponic_trays_total cannot be negative.")
        if self.field_fodder_area_sqm < 0:
            raise ValueError("field_fodder_area_sqm cannot be negative.")
        if self.daily_water_limit_l <= 0:
            raise ValueError("daily_water_limit_l must be positive.")


@dataclass
class DailyFarmAction:
    """
    Management decisions executed on simulation day t.
    
    Attributes:
        trays_to_seed: Number of empty trays washed, soaked, and seeded today (integer).
        harvest_field_today: Boolean flag indicating whether to cut the ready field plot today.
        feed_allocation_kg_dm: Dictionary mapping feed name -> Dry Matter kg offered to the pen today.
    """
    trays_to_seed: int = 4
    harvest_field_today: bool = False
    feed_allocation_kg_dm: Dict[str, float] = field(default_factory=dict)


@dataclass
class DailyStepRecord:
    """Complete telemetry record for one simulation step."""
    day: int
    mean_body_weight_kg: float
    pen_total_weight_kg: float
    daily_adg_g_head: float
    dmi_capacity_kg_dm: float
    dmi_consumed_kg_dm: float
    me_consumed_mj: float
    me_maintenance_required_mj: float
    cp_consumed_g: float
    cp_required_g: float
    dietary_ndf_pct: float
    daily_feed_cost: float
    daily_water_cost: float
    cash_balance: float
    water_used_liters: float
    hydroponic_harvest_kg_fresh: float
    field_harvest_kg_fresh: float
    stored_green_kg_dm: float
    stored_roughage_kg_dm: float
    stored_concentrate_kg_dm: float
    spoilage_loss_kg_dm: float
    safety_violation_acidosis: bool
    safety_violation_energy_deficit: bool


@dataclass
class SimulationConfig:
    """Configurable hyperparameters for simulation experiment."""
    horizon_days: int = 60
    daily_step_dt: float = 1.0
    random_seed: int = 42
    ambient_temp_c_mean: float = 25.0
    ambient_temp_c_amplitude: float = 5.0
    water_scarcity_factor: float = 1.0


@dataclass
class SimulationResult:
    """Encapsulates full simulation trajectory and summary diagnostics."""
    config: SimulationConfig
    pen_profile: GoatPenProfile
    resources: FarmResources
    trajectory_df: pd.DataFrame
    total_feed_cost: float
    total_water_cost: float
    initial_pen_weight_kg: float
    final_pen_weight_kg: float
    total_pen_weight_gain_kg: float
    gross_margin_over_feed_cost: float
    total_spoilage_loss_kg_dm: float
    acidosis_risk_days: int
    energy_deficit_days: int
    execution_status: str = "COMPLETED"
    notes: str = ""


def validate_feed_library_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates a pandas DataFrame containing a feed library against required schemas.
    Returns (is_valid, list_of_error_messages).
    """
    required_cols = [
        "name",
        "category",
        "dry_matter_pct",
        "crude_protein_pct_dm",
        "metabolizable_energy_mj_kg_dm",
        "ndf_pct_dm",
        "unit_cost_as_fed",
    ]
    errors = []
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    for idx, row in df.iterrows():
        try:
            FeedItem(
                name=str(row["name"]),
                category=str(row["category"]),
                dry_matter_pct=float(row["dry_matter_pct"]),
                crude_protein_pct_dm=float(row["crude_protein_pct_dm"]),
                metabolizable_energy_mj_kg_dm=float(row["metabolizable_energy_mj_kg_dm"]),
                ndf_pct_dm=float(row["ndf_pct_dm"]),
                unit_cost_as_fed=float(row["unit_cost_as_fed"]),
                provenance_source=str(row.get("provenance_source", "CSV_Upload")),
                evidence_status=str(row.get("evidence_status", "unverified")),
                notes=str(row.get("notes", "")),
            )
        except Exception as e:
            errors.append(f"Row {idx} ('{row.get('name', 'unknown')}'): {str(e)}")

    return len(errors) == 0, errors


def load_feed_library_from_csv(csv_path: str) -> Dict[str, FeedItem]:
    """
    Loads and strictly validates feed items from a CSV file.
    Raises ValueError with descriptive errors if validation fails.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feed library CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    is_valid, errors = validate_feed_library_df(df)
    if not is_valid:
        error_msg = "\n".join(errors[:10])
        raise ValueError(f"Feed library failed validation rules:\n{error_msg}")

    library: Dict[str, FeedItem] = {}
    for _, row in df.iterrows():
        item = FeedItem(
            name=str(row["name"]),
            category=str(row["category"]),
            dry_matter_pct=float(row["dry_matter_pct"]),
            crude_protein_pct_dm=float(row["crude_protein_pct_dm"]),
            metabolizable_energy_mj_kg_dm=float(row["metabolizable_energy_mj_kg_dm"]),
            ndf_pct_dm=float(row["ndf_pct_dm"]),
            unit_cost_as_fed=float(row["unit_cost_as_fed"]),
            provenance_source=str(row.get("provenance_source", "Sourced CSV")),
            evidence_status=str(row.get("evidence_status", "sourced")),
            notes=str(row.get("notes", "")),
        )
        library[item.name] = item

    return library

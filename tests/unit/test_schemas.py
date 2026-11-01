"""
Unit tests for data schemas, unit ranges, and CSV validation in models/schemas.py.
"""

import pytest
import pandas as pd
from models.schemas import FeedItem, GoatPenProfile, FarmResources, validate_feed_library_df


def test_valid_feed_item_creation():
    item = FeedItem(
        name="Test_Hay",
        category="dry_roughage",
        dry_matter_pct=88.0,
        crude_protein_pct_dm=10.0,
        metabolizable_energy_mj_kg_dm=8.5,
        ndf_pct_dm=60.0,
        unit_cost_as_fed=5.0,
        provenance_source="Test Literature",
        evidence_status="sourced",
    )
    assert item.name == "Test_Hay"
    assert item.cost_per_kg_dm == pytest.approx(5.0 / 0.88, rel=1e-3)


def test_feed_item_invalid_ranges_raise_error():
    with pytest.raises(ValueError, match="dry_matter_pct"):
        FeedItem(
            name="Bad_DM",
            category="dry_roughage",
            dry_matter_pct=105.0,  # Impossible DM > 100%
            crude_protein_pct_dm=10.0,
            metabolizable_energy_mj_kg_dm=8.0,
            ndf_pct_dm=50.0,
            unit_cost_as_fed=2.0,
        )

    with pytest.raises(ValueError, match="Invalid category"):
        FeedItem(
            name="Bad_Cat",
            category="non_existent_category",
            dry_matter_pct=80.0,
            crude_protein_pct_dm=10.0,
            metabolizable_energy_mj_kg_dm=8.0,
            ndf_pct_dm=50.0,
            unit_cost_as_fed=2.0,
        )


def test_pen_profile_validation():
    profile = GoatPenProfile(head_count=5, initial_mean_weight_kg=20.0)
    assert profile.head_count == 5
    assert profile.initial_mean_weight_kg == 20.0

    with pytest.raises(ValueError, match="head_count must be at least 1"):
        GoatPenProfile(head_count=0)


def test_validate_feed_library_df():
    valid_data = {
        "name": ["FeedA"],
        "category": ["concentrate"],
        "dry_matter_pct": [90.0],
        "crude_protein_pct_dm": [18.0],
        "metabolizable_energy_mj_kg_dm": [11.5],
        "ndf_pct_dm": [30.0],
        "unit_cost_as_fed": [15.0],
    }
    df = pd.DataFrame(valid_data)
    is_valid, errors = validate_feed_library_df(df)
    assert is_valid
    assert len(errors) == 0

    # Test missing required column
    invalid_df = df.drop(columns=["metabolizable_energy_mj_kg_dm"])
    is_valid_inv, errors_inv = validate_feed_library_df(invalid_df)
    assert not is_valid_inv
    assert "Missing required columns" in errors_inv[0]

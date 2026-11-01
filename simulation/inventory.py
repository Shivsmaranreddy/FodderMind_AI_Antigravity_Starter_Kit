"""
inventory.py - FIFO feed inventory management and aerobic spoilage dynamics.

Tracks stored Dry Matter (DM) with category-dependent shelf lives:
- Green fodder: Highly perishable (spoilage within 3 days without preservation).
- Dry roughage and concentrates: Stable dry storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FeedBatch:
    """Represents a specific batch of feed in storage."""
    feed_name: str
    category: str
    dry_matter_kg: float
    age_days: int = 0
    max_shelf_life_days: int = 3  # default for green fodder


class FeedInventory:
    """
    Manages farm feed inventory across all storage categories.
    Implements FIFO consumption and time-dependent spoilage.
    """
    def __init__(self):
        # Dictionary mapping feed_name -> List of FeedBatch
        self.batches: Dict[str, List[FeedBatch]] = {}

    def add_stock(
        self,
        feed_name: str,
        category: str,
        dry_matter_kg: float,
        shelf_life_days: int = 60,
    ):
        """Adds a fresh batch of feed to storage."""
        if dry_matter_kg <= 0.0:
            return

        # Override shelf life for high-moisture green feeds
        if "green" in category.lower():
            shelf_life_days = 3

        if feed_name not in self.batches:
            self.batches[feed_name] = []

        self.batches[feed_name].append(
            FeedBatch(
                feed_name=feed_name,
                category=category,
                dry_matter_kg=dry_matter_kg,
                age_days=0,
                max_shelf_life_days=shelf_life_days,
            )
        )

    def get_available_dm(self, feed_name: str) -> float:
        """Returns total unexpired dry matter available for a specific feed."""
        if feed_name not in self.batches:
            return 0.0
        return sum(b.dry_matter_kg for b in self.batches[feed_name])

    def withdraw_stock(self, feed_name: str, requested_dm_kg: float) -> float:
        """
        Withdraws dry matter using First-In-First-Out (FIFO) ordering.
        Returns the actual dry matter withdrawn (may be less than requested if stock is low).
        """
        if feed_name not in self.batches or requested_dm_kg <= 0.0:
            return 0.0

        withdrawn = 0.0
        remaining_needed = requested_dm_kg

        for batch in self.batches[feed_name]:
            if batch.dry_matter_kg <= remaining_needed:
                withdrawn += batch.dry_matter_kg
                remaining_needed -= batch.dry_matter_kg
                batch.dry_matter_kg = 0.0
            else:
                batch.dry_matter_kg -= remaining_needed
                withdrawn += remaining_needed
                remaining_needed = 0.0
                break

        # Remove exhausted batches
        self.batches[feed_name] = [b for b in self.batches[feed_name] if b.dry_matter_kg > 1e-4]
        return withdrawn

    def step_day(self) -> float:
        """
        Advances age of all batches by 1 day and eliminates spoiled feed.
        Returns total dry matter mass lost to spoilage today (kg DM).
        """
        total_spoiled_dm = 0.0

        for feed_name, batch_list in self.batches.items():
            active_batches = []
            for b in batch_list:
                b.age_days += 1
                if b.age_days > b.max_shelf_life_days:
                    total_spoiled_dm += b.dry_matter_kg
                else:
                    active_batches.append(b)
            self.batches[feed_name] = active_batches

        return total_spoiled_dm

    def get_category_totals(self) -> Tuple[float, float, float]:
        """
        Returns (stored_green_dm_kg, stored_roughage_dm_kg, stored_concentrate_dm_kg).
        """
        green = 0.0
        roughage = 0.0
        concentrate = 0.0

        for feed_name, batch_list in self.batches.items():
            for b in batch_list:
                cat = b.category.lower()
                if "green" in cat:
                    green += b.dry_matter_kg
                elif "roughage" in cat or "hay" in cat or "stover" in cat:
                    roughage += b.dry_matter_kg
                else:
                    concentrate += b.dry_matter_kg

        return green, roughage, concentrate

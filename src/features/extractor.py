"""Feature extraction dari cleaned data."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.geo import haversine_vectorized


class FeatureExtractor:
    """Extract base features + derivative features untuk lokasi (lat, lng)."""

    def __init__(self, cafes, owner, poi):
        self.cafes = cafes
        self.owner = owner
        self.poi = poi

    def count_in_radius(self, lat, lng, df, radius_m) -> int:
        """Hitung radius."""
        if df.empty:
            return 0
        d = haversine_vectorized(lat, lng, df["lat"].values, df["lng"].values)
        return int((d < radius_m).sum())

    def competitor_count(self, lat, lng, self_id, max_radius) -> dict:
        """Stats cafe lain dalam radius tertentu"""
        others = (
            self.cafes[self.cafes["place_id"] != self_id]
            if self_id is not None
            else self.cafes
        )
        if others.empty:
            return {"n_comp": 0, "avg_rating": 0, "total_reviews": 0, "max_reviews": 0}

        d = haversine_vectorized(lat, lng, others["lat"].values, others["lng"].values)
        subset = others[d < max_radius]

        if subset.empty:
            return {"n_comp": 0, "avg_rating": 0, "total_reviews": 0, "max_reviews": 0}

        return {
            "n_comp": len(subset),
            "avg_rating": round(float(subset["rating"].mean()), 1),
            "total_reviews": int(subset["reviews_count"].sum()),
            "max_reviews": int(subset["reviews_count"].max()),
        }

    def nearest_owner(self, lat, lng) -> float | None:
        """Jarak ke toko owner terdekat"""
        if self.owner.empty:
            return None
        d = haversine_vectorized(
            lat, lng, self.owner["lat"].values, self.owner["lng"].values
        )
        return float(d.min())

    def _poi(self, category) -> pd.DataFrame:
        return self.poi.get(category, pd.DataFrame())

    def extract_features(self, lat, lng, self_id=None) -> dict:
        """Extract semua features"""
        n_comp_500 = self.competitor_count(lat, lng, self_id, 500)
        n_comp_2km = self.competitor_count(lat, lng, self_id, 2000)
        nearest = self.nearest_owner(lat, lng)

        feats = {
            # Kompetisi
            "n_competitors_500m": n_comp_500["n_comp"],
            "avg_competitor_rating_500": n_comp_500["avg_rating"],
            "total_competitor_reviews_500": n_comp_500["total_reviews"],
            "max_competitor_reviews_500": n_comp_500["max_reviews"],
            "n_competitors_2km": n_comp_2km["n_comp"],
            "avg_competitor_rating_2km": n_comp_2km["avg_rating"],
            "total_competitor_reviews_2km": n_comp_2km["total_reviews"],
            "max_competitor_reviews_2km": n_comp_2km["max_reviews"],
            # Cannibalizatio
            "nearest_owner_store_m": round(nearest, 2) if nearest is not None else None,
            # POI counts
            "n_offices_500m": self.count_in_radius(lat, lng, self._poi("office"), 500),
            "n_offices_2km": self.count_in_radius(lat, lng, self._poi("office"), 2000),
            "n_malls_500m": self.count_in_radius(lat, lng, self._poi("mall"), 500),
            "n_malls_2km": self.count_in_radius(lat, lng, self._poi("mall"), 2000),
            "n_transit_500m": self.count_in_radius(lat, lng, self._poi("transit"), 500),
            "n_transit_2km": self.count_in_radius(lat, lng, self._poi("transit"), 2000),
            "n_schools_500m": self.count_in_radius(lat, lng, self._poi("school"), 500),
            "n_schools_2km": self.count_in_radius(lat, lng, self._poi("school"), 2000),
        }

        # Derivative features
        feats["density_ratio_500m_2km"] = (
            feats["n_competitors_500m"] / (feats["n_competitors_2km"] + 1)
        )
        feats["avg_reviews_per_cafe_2km"] = (
            feats["total_competitor_reviews_2km"] / (feats["n_competitors_2km"] + 1)
        )
        feats["market_saturation"] = (
            feats["n_competitors_500m"] * feats["avg_competitor_rating_2km"]
        )
        feats["office_transit_combo"] = (
            feats["n_offices_500m"] * feats["n_transit_500m"]
        )
        feats["has_strong_competitor"] = int(feats["max_competitor_reviews_2km"] > 1000)
        feats["anchor_score"] = (
            feats["n_malls_2km"] * 0.5 + feats["n_transit_500m"] * 1.0
        )

        return feats

    def build_training_dataset(self) -> pd.DataFrame:
        """Build training dataset"""
        print(f"Building training dataset from {len(self.cafes)} cafes...")

        rows = []
        for _, row in tqdm(self.cafes.iterrows(), total=len(self.cafes), desc="Extracting features"):
            self_id = row.get("place_id", row.get("name", ""))
            feats = self.extract_features(row["lat"], row["lng"], self_id=self_id)

            feats["target"] = row["rating"] * np.log1p(row["reviews_count"])
            feats["cafe_id"] = self_id
            feats["lat"], feats["lng"] = row["lat"], row["lng"]
            feats["rating"] = row["rating"]
            feats["reviews_count"] = row["reviews_count"]
    
            for col in ("kecamatan", "kota"):
                if col in self.cafes.columns:
                    feats[col] = row.get(col, "")

            rows.append(feats)

        df_train = pd.DataFrame(rows)
        print(df_train.columns.tolist())
        print(f"✓ Training dataset: {df_train.shape}")
        return df_train
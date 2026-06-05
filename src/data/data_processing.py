"""Clean data"""

import pandas as pd
from src.data.data_loader import ROOT
from src.data import supabase_io as sio


# Cache CSV lokal (opsional, di-gitignore). Sumber kebenaran = Supabase.
CLEAN_DIR = ROOT / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

class DataProcessor:
    """Clean raw data"""

    def __init__(self, data, jakarta_bbox, cafe_keywords):
        self.cafe_keywords = cafe_keywords
        self.jakarta_bbox = jakarta_bbox
        self.cafes = self.clean_cafes(data["cafes"])
        self.owner = self.clean_owner(data["owner"])
        self.poi = self.clean_poi(data["poi"])

    def filter_jakarta(self, df, lat_col="lat", lng_col="lng"):
        return (
            df[lat_col].between(self.jakarta_bbox["lat_min"], self.jakarta_bbox["lat_max"])
            & df[lng_col].between(self.jakarta_bbox["lng_min"], self.jakarta_bbox["lng_max"])
        )

    def remove_duplicate_coords(self, df, decimals=5):
        df = df.copy()
        df["lat_round"] = df["lat"].round(decimals)
        df["lng_round"] = df["lng"].round(decimals)

        df = df.drop_duplicates(
            subset=["lat_round", "lng_round"]
        )
        return df

    def rename_columns(self, df):
        if "latitude" in df.columns:
            df = df.rename(columns={"latitude": "lat"})
        if "longitude" in df.columns:
            df = df.rename(columns={"longitude": "lng"})
        if "total_reviews" in df.columns:
            df = df.rename(columns={"total_reviews": "reviews_count"})
        if "user_rating_count" in df.columns:
            df = df.rename(columns={"user_rating_count": "reviews_count"})
        return df

    def clean_cafes(self, df):
        """Clean cafes Google Maps"""
        df = df.copy()
        n0 = len(df)

        df = self.rename_columns(df)
        df = df[df["lat"].notna() & df["lng"].notna()]
        df = df[df["rating"].notna()]
        df["reviews_count"] = df["reviews_count"].fillna(0).astype(int)
        df = df[self.filter_jakarta(df)]

        # Filter cafe-only
        pattern = "|".join(self.cafe_keywords)
        mask = pd.Series(False, index=df.index)
        if "primary_type" in df.columns:
            mask |= df["primary_type"].fillna("").str.lower().str.contains(pattern)
        if "types" in df.columns:
            mask |= df["types"].fillna("").str.lower().str.contains(pattern)
        mask |= df["name"].fillna("").str.lower().str.contains(pattern)
        df = df[mask]

        df = self.remove_duplicate_coords(df)

        print(f"  ✓ cafes: {len(df)} (from {n0})")
        return df

    def clean_owner(self, df):
        """Clean owner stores"""
        df = df.copy()
        n0 = len(df)

        df = self.rename_columns(df)
        df = df[df["lat"].notna() & df["lng"].notna()]
        df = df[self.filter_jakarta(df)]

        if "tanggal_buka" in df.columns:
            df["tanggal_buka"] = pd.to_datetime(df["tanggal_buka"], errors="coerce")
            df["umur_bulan"] = (
                (pd.Timestamp.now() - df["tanggal_buka"]).dt.days / 30
            ).round(1)

        df = df.reset_index(drop=True)
        print(f"  ✓ owner: {len(df)} (from {n0})")
        return df

    def clean_poi(self, poi_dict):
        """Clean semua POI kategori."""
        result = {}
        for cat, df in poi_dict.items():
            df = df.copy()
            n0 = len(df)

            df = self.rename_columns(df)
            df = df[df["lat"].notna() & df["lng"].notna()]
            df = df[self.filter_jakarta(df)]
            df = self.remove_duplicate_coords(df)

            drop_cols = ["all_tags"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

            df["category"] = cat
            df = df.reset_index(drop=True)

            result[cat] = df
            print(f"  ✓ poi {cat}: {len(df)} (from {n0})")

        return result

    def save(self, to_supabase=True, to_local=True):
        """Save semua clean data ke Supabase (utama) + cache CSV lokal (opsional)."""
        if to_supabase:
            print("\n↑ Upload clean data ke Supabase...")
            sio.upload_df(sio.CLEAN_CAFES, self.cafes)
            sio.upload_df(sio.CLEAN_OWNER, self.owner)
            for cat, df in self.poi.items():
                sio.upload_df(sio.clean_poi(cat), df)

        if to_local:
            self.cafes.to_csv(CLEAN_DIR / "cafes_clean.csv", index=False)
            self.owner.to_csv(CLEAN_DIR / "owner_clean.csv", index=False)
            for cat, df in self.poi.items():
                df.to_csv(CLEAN_DIR / f"osm_{cat}_clean.csv", index=False)
            print(f"\n✓ Cache lokal: {CLEAN_DIR.relative_to(ROOT)}/")

        print(f"  cafes_clean   : {len(self.cafes):>6,} rows")
        print(f"  owner_clean   : {len(self.owner):>6,} rows")
        for cat, df in self.poi.items():
            print(f"  clean_poi_{cat}{' ' * (8 - len(cat))} : {len(df):>6,} rows")

    def check_owner_overlap(self, threshold_m=100):
        """Cek apakah owner store muncul juga di cafes"""
        try:
            from src.utils.geo import haversine_vectorized
        except ImportError:
            print("  ⚠️  src.utils.geo not available, skip")
            return []

        matches = []
        for _, owner_row in self.owner.iterrows():
            d = haversine_vectorized(
                owner_row["lat"], owner_row["lng"],
                self.cafes["lat"].values, self.cafes["lng"].values,
            )
            nearest_idx = d.argmin()
            nearest_dist = d[nearest_idx]

            if nearest_dist < threshold_m:
                matches.append({
                    "owner_name": owner_row.get("nama", "Unknown"),
                    "gmaps_name": self.cafes.iloc[nearest_idx].get("name", "Unknown"),
                    "distance_m": round(float(nearest_dist), 1),
                })

        if matches:
            print(f"\n  ⚠️  {len(matches)} owner store overlap (< {threshold_m}m):")
            for m in matches:
                print(f"    - {m['owner_name']} ↔ {m['gmaps_name']} ({m['distance_m']}m)")
        else:
            print(f"\n  ✓ No overlap < {threshold_m}m")

        return matches
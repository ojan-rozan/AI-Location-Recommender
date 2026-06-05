"""
Seed Supabase dari CSV lokal
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import supabase_io as sio

RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
HELPER = ROOT / "data" / "helper"


def data_csv(path: Path):
    if not path.exists():
        print(f"  ⚠️  skip (gak ada): {path.relative_to(ROOT)}")
        return None
    return pd.read_csv(path)


def push(dataset: str, df):
    if df is None or df.empty:
        return
    sio.upload_df(dataset, df)


def main():
    print("=== Seed Supabase dari CSV lokal ===\n")

    # Helper / reference
    push(sio.OWNER_STORES, data_csv(HELPER / "owner_stores.csv"))
    push(sio.KECAMATAN_REF, data_csv(HELPER / "kecamatan_jakarta.csv"))

    # Raw scraping results
    push(sio.RAW_CAFES, data_csv(RAW / "places_cafe.csv"))
    for cat in sio.POI_CATEGORIES:
        push(sio.raw_poi(cat), data_csv(RAW / f"osm_{cat}_jakarta.csv"))

    # Clean data (kalau udah ada)
    push(sio.CLEAN_CAFES, data_csv(CLEAN / "cafes_clean.csv"))
    push(sio.CLEAN_OWNER, data_csv(CLEAN / "owner_clean.csv"))
    for cat in sio.POI_CATEGORIES:
        push(sio.clean_poi(cat), data_csv(CLEAN / f"osm_{cat}_clean.csv"))

    print("\n✅ Selesai. Data lokal sekarang ada di Supabase.")


if __name__ == "__main__":
    main()

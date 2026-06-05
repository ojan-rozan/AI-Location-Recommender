"""
Supabase data IO
"""

import os
import json

import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

RAW_CAFES = "raw_cafes"
OWNER_STORES = "owner_stores"
KECAMATAN_REF = "kecamatan_ref"
POI_CATEGORIES = ["office", "mall", "transit", "school"]
CLEAN_CAFES = "clean_cafes"
CLEAN_OWNER = "clean_owner"


def raw_poi(category: str):
    return f"raw_poi_{category}"


def clean_poi(category: str):
    return f"clean_poi_{category}"


def all_tables():
    """create name table"""
    names = [RAW_CAFES, OWNER_STORES, KECAMATAN_REF, CLEAN_CAFES, CLEAN_OWNER]
    for cat in POI_CATEGORIES:
        names.append(raw_poi(cat))
        names.append(clean_poi(cat))
    return names


_client = None


def get_client():
    """Import Supabase client"""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL + SUPABASE_KEY wajib di-set di .env "
            )
        _client = create_client(url, key)
    return _client


def _to_json_records(df):
    """
    DataFrame -> list of dict yang aman buat JSON.
    Lewat df.to_json: Timestamp/date -> ISO string, NaN/NaT -> null,
    tipe numpy -> tipe Python native.
    """
    return json.loads(df.to_json(orient="records", date_format="iso"))


def upload_df(table, df, chunk_size=500, replace=True):
    """
    Upload DataFrame ke tabel Supabase
    """
    client = get_client()

    if replace:
        # hapus semua baris (id selalu >= 1)
        client.table(table).delete().gte("id", 0).execute()

    records = _to_json_records(df)
    rows = [{"data": r} for r in records]

    for i in range(0, len(rows), chunk_size):
        client.table(table).insert(rows[i:i + chunk_size]).execute()

    print(f"  ↑ supabase '{table}': {len(rows)} rows uploaded")
    return len(rows)


def read_df(table: str) -> pd.DataFrame:
    """
    Read tabel Supabase `table` jadi DataFrame
    """
    client = get_client()
    payloads = []
    page_size = 1000
    page = 0

    while True:
        start = page * page_size
        end = start + page_size - 1
        resp = (
            client.table(table)
            .select("data")
            .range(start, end)
            .execute()
        )
        batch = resp.data or []
        payloads.extend([row["data"] for row in batch])
        if len(batch) < page_size:
            break
        page += 1

    df = pd.DataFrame(payloads)
    print(f"  ✓ supabase '{table}': {len(df)} rows")
    return df

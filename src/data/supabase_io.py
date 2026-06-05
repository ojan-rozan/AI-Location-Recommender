"""
Supabase data IO via REST API.

Sengaja pakai requests daripada library supabase karena lebih stabil

Struktur tabel:
- id
- data (jsonb)
- created_at
"""

import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Dataset names
RAW_CAFES = "raw_cafes"
OWNER_STORES = "owner_stores"
KECAMATAN_REF = "kecamatan_ref"

POI_CATEGORIES = [
    "office",
    "mall",
    "transit",
    "school",
]

CLEAN_CAFES = "clean_cafes"
CLEAN_OWNER = "clean_owner"

session = requests.Session()


def raw_poi(category):
    return f"raw_poi_{category}"


def clean_poi(category):
    return f"clean_poi_{category}"


def all_tables():
    tables = [RAW_CAFES, OWNER_STORES, KECAMATAN_REF, CLEAN_CAFES, CLEAN_OWNER]

    for category in POI_CATEGORIES:
        tables.extend([
            raw_poi(category),
            clean_poi(category),
        ])

    return tables


def get_credentials():
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_KEY", "").strip()

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL dan SUPABASE_KEY wajib diisi"
        )

    return url, key


def get_headers(key):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }


def get_write_headers(key):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def dataframe_to_records(df):
    """
    Convert DataFrame menjadi list of dict yang JSON-safe.
    Timestamp -> ISO string
    NaN / NaT -> null
    """
    return json.loads(
        df.to_json(
            orient="records",
            date_format="iso",
        )
    )


def upload_df(table, df, chunk_size=500, replace=True):
    """
    Upload DataFrame ke Supabase.
    """

    if df is None or df.empty:
        print(f"  ⚠️ skip '{table}' (empty dataframe)")
        return 0

    url, key = get_credentials()

    endpoint = f"{url}/rest/v1/{table}"
    headers = get_write_headers(key)

    if replace:
        response = session.delete(
            f"{endpoint}?id=gte.0",
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()

    records = dataframe_to_records(df)
    rows = [{"data": row} for row in records]

    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]

        response = session.post(
            endpoint,
            headers=headers,
            json=chunk,
            timeout=120,
        )

        response.raise_for_status()

    print(f"  ↑ {table}: {len(rows):,} rows uploaded")

    return len(rows)


def read_df(table):
    """
    Read seluruh isi tabel Supabase menjadi DataFrame.
    """

    url, key = get_credentials()

    endpoint = f"{url}/rest/v1/{table}"
    headers = get_headers(key)

    page_size = 1000
    offset = 0

    records = []

    while True:
        response = session.get(
            endpoint,
            headers=headers,
            params={
                "select": "data",
                "limit": page_size,
                "offset": offset,
            },
            timeout=60,
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        records.extend(
            row["data"]
            for row in batch
        )

        if len(batch) < page_size:
            break

        offset += page_size

    df = pd.DataFrame(records)

    print(f"  ✓ {table}: {len(df):,} rows")

    return df
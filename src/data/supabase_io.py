"""
Supabase data IO via REST API (pakai requests / HTTP-1.1).

Sengaja TIDAK pakai library `supabase` (yang pakai httpx HTTP/2) karena
HTTP/2 sering ke-reset (StreamReset) di sebagian jaringan (mis. Hugging Face).
requests pakai HTTP/1.1 -> stabil.

Tiap dataset = 1 tabel sendiri (kolom: id, data jsonb, created_at).
Butuh di env: SUPABASE_URL, SUPABASE_KEY (service_role untuk tulis).
"""

import os
import json

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Nama tabel/dataset
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
    names = [RAW_CAFES, OWNER_STORES, KECAMATAN_REF, CLEAN_CAFES, CLEAN_OWNER]
    for cat in POI_CATEGORIES:
        names.append(raw_poi(cat))
        names.append(clean_poi(cat))
    return names


def _creds():
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL + SUPABASE_KEY wajib di-set di .env")
    return url, key


def _headers(key, write=False):
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    if write:
        h["Content-Type"] = "application/json"
        h["Prefer"] = "return=minimal"
    return h


def _to_json_records(df):
    """DataFrame -> list dict JSON-safe (Timestamp->ISO, NaN/NaT->null)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def upload_df(table, df, chunk_size=500, replace=True):
    """Upload DataFrame ke tabel Supabase via REST (1 row = 1 payload jsonb)."""
    url, key = _creds()
    base = f"{url}/rest/v1/{table}"

    if replace:
        # hapus semua baris (id selalu >= 1)
        r = requests.delete(f"{base}?id=gte.0", headers=_headers(key, write=True), timeout=60)
        r.raise_for_status()

    records = _to_json_records(df)
    rows = [{"data": rec} for rec in records]

    for i in range(0, len(rows), chunk_size):
        r = requests.post(base, headers=_headers(key, write=True),
                          json=rows[i:i + chunk_size], timeout=120)
        r.raise_for_status()

    print(f"  ↑ supabase '{table}': {len(rows)} rows uploaded")
    return len(rows)


def read_df(table: str) -> pd.DataFrame:
    """Read tabel Supabase via REST jadi DataFrame (auto-paginate)."""
    url, key = _creds()
    base = f"{url}/rest/v1/{table}"
    payloads = []
    offset = 0
    page = 1000

    while True:
        params = {"select": "data", "limit": page, "offset": offset}
        r = requests.get(base, headers=_headers(key), params=params, timeout=60)
        r.raise_for_status()
        batch = r.json()
        payloads.extend([row["data"] for row in batch])
        if len(batch) < page:
            break
        offset += page

    df = pd.DataFrame(payloads)
    print(f"  ✓ supabase '{table}': {len(df)} rows")
    return df

"""
Supabase data IO
"""

import os
import json
import time

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
        url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
        key = os.environ.get("SUPABASE_KEY", "").strip()
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL + SUPABASE_KEY wajib di-set di .env "
            )
        _client = create_client(url, key)
    return _client


def _reset_client():
    """Buang client lama (koneksi ke-reset) biar bikin koneksi baru."""
    global _client
    _client = None


def _with_retry(fn, tries=4, delay=1.5):
    """
    Jalanin fn() dengan retry. Supabase/httpx kadang lempar StreamReset
    (HTTP/2 reset) yang transient — coba ulang dengan koneksi baru.
    """
    last_err = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            print(f"  ⚠️  supabase error (attempt {attempt + 1}/{tries}): "
                  f"{type(e).__name__}: {str(e)[:80]}")
            _reset_client()
            time.sleep(delay)
    raise last_err


def _to_json_records(df):
    """
    DataFrame -> list of dict yang aman buat JSON.
    Lewat df.to_json: Timestamp/date -> ISO string, NaN/NaT -> null,
    tipe numpy -> tipe Python native.
    """
    return json.loads(df.to_json(orient="records", date_format="iso"))


def upload_df(table, df, chunk_size=500, replace=True):
    """
    Upload DataFrame ke tabel Supabase (dengan retry kalau koneksi reset).
    """
    records = _to_json_records(df)
    rows = [{"data": r} for r in records]

    def _do():
        client = get_client()
        if replace:
            client.table(table).delete().gte("id", 0).execute()
        for i in range(0, len(rows), chunk_size):
            client.table(table).insert(rows[i:i + chunk_size]).execute()
        return len(rows)

    n = _with_retry(_do)
    print(f"  ↑ supabase '{table}': {n} rows uploaded")
    return n


def read_df(table: str) -> pd.DataFrame:
    """
    Read tabel Supabase `table` jadi DataFrame (dengan retry kalau reset).
    """
    def _do():
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
        return payloads

    payloads = _with_retry(_do)
    df = pd.DataFrame(payloads)
    print(f"  ✓ supabase '{table}': {len(df)} rows")
    return df

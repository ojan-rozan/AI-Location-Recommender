"""
REST API — Coffee Shop Location Recommender (FastAPI).

Semua logika inferensi langsung di file ini (load model + data dari Supabase,
predict, nearest, map-data, analyze).

Jalanin dari project root:
    uvicorn app.api:app --reload --port 8000

Docs interaktif (Swagger): http://localhost:8000/docs
"""

import sys
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.geo import haversine_vectorized
from src.features.extractor import FeatureExtractor
from src.models.model import DemandModel
from src.data import supabase_io as sio


# helpers
def _json_safe(obj):
    """Coerce numpy/NaN -> tipe JSON-safe lewat pandas."""
    return json.loads(pd.Series(obj).to_json())


def _records(df, cols=None, limit=None):
    if df is None or df.empty:
        return []
    d = df
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    if limit:
        d = d.head(limit)
    return json.loads(d.to_json(orient="records"))


# resource loader (singleton)
class _Resources:
    """Holder model + data. Load sekali, cache di memori."""

    def __init__(self):
        self.model = None
        self.cafes = None
        self.owner = None
        self.poi = None
        self.extractor = None
        self.retriever = None
        self.generator = None

    def load(self):
        self.model = DemandModel.load(ROOT / "models" / "xgb_demand.pkl")

        self.cafes = sio.read_df(sio.CLEAN_CAFES)
        self.owner = sio.read_df(sio.CLEAN_OWNER)
        self.poi = {}
        for cat in sio.POI_CATEGORIES:
            df = sio.read_df(sio.clean_poi(cat))
            self.poi[cat] = df if not df.empty else pd.DataFrame(columns=["lat", "lng", "name"])

        self.extractor = FeatureExtractor(cafes=self.cafes, owner=self.owner, poi=self.poi)
        self.retriever = self._try_retriever()
        self.generator = self._try_generator()
        return self

    @staticmethod
    def _try_retriever():
        try:
            from src.rag.retriever import KnowledgeRetriever
            return KnowledgeRetriever()
        except Exception as e:
            print(f"  ⚠️  retriever off: {e}")
            return None

    @staticmethod
    def _try_generator():
        try:
            from src.rag.generator import SummaryGenerator
            return SummaryGenerator()
        except Exception as e:
            print(f"  ⚠️  generator off: {e}")
            return None


_res = None


def res() -> _Resources:
    global _res
    if _res is None:
        _res = _Resources().load()
    return _res


# core logic
def _nearest_df(df, lat, lng, top_n, cols=None):
    if df is None or df.empty:
        return []
    d = df.copy()
    d["distance_m"] = haversine_vectorized(lat, lng, d["lat"].values, d["lng"].values)
    d = d.nsmallest(top_n, "distance_m")
    if cols:
        d = d[[c for c in cols if c in d.columns] + ["distance_m"]]
    return json.loads(d.to_json(orient="records"))


def do_predict(lat, lng):
    r = res()
    feats = r.extractor.extract_features(lat, lng)
    X = pd.DataFrame([{c: feats.get(c, 0) for c in r.model.feature_cols}])
    result = r.model.predict_with_explanation(X)[0]
    return _json_safe({
        "score": result["score_0_100"],
        "raw_prediction": result["raw_prediction"],
        "top_factors": result["top_factors"],
        **feats,
    })


def do_map_data(lat, lng, radius_m):
    r = res()

    def within(df, limit):
        if df is None or df.empty or "lat" not in df.columns:
            return df
        d = df.copy()
        d["_dist"] = haversine_vectorized(lat, lng, d["lat"].values, d["lng"].values)
        return d[d["_dist"] <= radius_m].nsmallest(limit, "_dist")

    return {
        "cafes": _records(within(r.cafes, 200),
                          cols=["name", "rating", "reviews_count", "lat", "lng"]),
        "owner": _records(r.owner,
                          cols=["nama", "tipe", "omzet_bulanan_juta", "lat", "lng"]),
        "poi": {
            cat: _records(within(r.poi.get(cat), 100), cols=["name", "lat", "lng"])
            for cat in (r.poi or {})
        },
    }


def do_analyze(lat, lng):
    r = res()
    ml_data = do_predict(lat, lng)

    docs, topics = [], []
    if r.retriever is not None:
        try:
            docs, topics = r.retriever.retrieve(ml_data)
        except Exception as e:
            print(f"  ⚠️  retrieve gagal: {e}")

    summary = None
    if r.generator is not None:
        try:
            summary = r.generator.generate(lat, lng, ml_data, docs)
        except Exception as e:
            print(f"  ⚠️  LLM gagal: {e}")

    return {
        "lat": lat, "lng": lng,
        "score": ml_data["score"],
        "top_factors": ml_data["top_factors"],
        "topics": topics, "n_docs": len(docs),
        "summary": summary, "ml_data": ml_data,
    }


def do_features(lat, lng):
    return _json_safe(res().extractor.extract_features(lat, lng))


def do_nearest(lat, lng, top_n=3):
    r = res()
    return {
        "competitors": _nearest_df(r.cafes, lat, lng, top_n,
                                   cols=["name", "rating", "reviews_count", "lat", "lng"]),
        "owner_stores": _nearest_df(r.owner, lat, lng, top_n),
    }


# FastAPI
app = FastAPI(
    title="Coffee Shop Location API",
    description="Prediksi kelayakan lokasi coffee shop di Jakarta (ML + RAG).",
    version="1.0.0",
)


class Coord(BaseModel):
    lat: float = Field(..., ge=-90, le=90, examples=[-6.2297])
    lng: float = Field(..., ge=-180, le=180, examples=[106.8195])


@app.on_event("startup")
def _warmup():
    res()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/features", tags=["inference"])
def features(lat: float = Query(..., ge=-90, le=90), lng: float = Query(..., ge=-180, le=180)):
    """Fitur lokasi (buat quick stats)."""
    try:
        return do_features(lat, lng)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nearest", tags=["inference"])
def nearest(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    top_n: int = Query(3, ge=1, le=20),
):
    """Cafe kompetitor & owner store terdekat."""
    try:
        return do_nearest(lat, lng, top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/map-data", tags=["inference"])
def map_data(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    radius_m: float = Query(2000, ge=100, le=10000),
):
    """Semua titik (cafe, owner, POI) buat di-plot peta."""
    try:
        return do_map_data(lat, lng, radius_m)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", tags=["inference"])
def predict(body: Coord):
    """ML score + faktor utama."""
    try:
        return do_predict(body.lat, body.lng)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", tags=["inference"])
def analyze(body: Coord):
    """Predict + retrieve knowledge (RAG) + ringkasan LLM."""
    try:
        return do_analyze(body.lat, body.lng)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

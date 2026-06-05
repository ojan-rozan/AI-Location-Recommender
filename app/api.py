import json
import logging
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import supabase_io as sio
from src.features.extractor import FeatureExtractor
from src.models.model import DemandModel
from src.utils.geo import haversine_vectorized

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Coffee Shop Location API",
    description="Prediksi kelayakan lokasi coffee shop di Jakarta",
    version="1.0.0",
)


class Coord(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)

JAKARTA_BBOX = {"lat": (-6.40, -6.05), "lng": (106.65, 107.00)}


def in_jakarta(lat, lng):
    lo_lat, hi_lat = JAKARTA_BBOX["lat"]
    lo_lng, hi_lng = JAKARTA_BBOX["lng"]
    return lo_lat <= lat <= hi_lat and lo_lng <= lng <= hi_lng


@app.on_event("startup")
def load_resources():
    logger.info("Loading resources...")

    app.state.model = DemandModel.load(ROOT / "models" / "xgb_demand.pkl")

    app.state.cafes = sio.read_df(sio.CLEAN_CAFES)
    app.state.owner = sio.read_df(sio.CLEAN_OWNER)

    poi = {}

    for category in sio.POI_CATEGORIES:
        df = sio.read_df(sio.clean_poi(category))

        if df.empty:
            df = pd.DataFrame(columns=["lat", "lng", "name"])

        poi[category] = df

    app.state.poi = poi

    app.state.extractor = FeatureExtractor(cafes=app.state.cafes, owner=app.state.owner, poi=app.state.poi)

    try:
        from src.rag.retriever import KnowledgeRetriever

        app.state.retriever = KnowledgeRetriever()

    except Exception:
        logger.exception("Failed loading retriever")
        app.state.retriever = None

    try:
        from src.rag.generator import SummaryGenerator

        app.state.generator = SummaryGenerator()

    except Exception:
        logger.exception("Failed loading generator")
        app.state.generator = None


def add_distance_column(df, lat, lng):
    data = df.copy()
    data["distance_m"] = haversine_vectorized(lat, lng,data["lat"].values,data["lng"].values)
    return data


def nearest_records(df, lat, lng, top_n=3, columns=None):
    if df is None or df.empty:
        return []

    data = add_distance_column(df, lat, lng)
    data = data.nsmallest(top_n, "distance_m")

    if columns:
        keep_cols = [c for c in columns if c in data.columns]
        keep_cols.append("distance_m")
        data = data[keep_cols]

    return json.loads(data.to_json(orient="records"))


def nearby_records(df, lat, lng, radius_m, limit=100):
    if df is None or df.empty:
        return pd.DataFrame()

    data = add_distance_column(df, lat, lng)
    data = data[data["distance_m"] <= radius_m]

    return data.nsmallest(limit, "distance_m")


def extract_features(lat, lng):
    return app.state.extractor.extract_features(lat, lng)


def predict_location(lat, lng):
    features = extract_features(lat, lng)

    X = pd.DataFrame([
        {col: features.get(col, 0) for col in app.state.model.feature_cols}
    ])

    result = app.state.model.predict_with_explanation(X)[0]

    return {
        "score": result["score_0_100"],
        "raw_prediction": result["raw_prediction"],
        "top_factors": result["top_factors"],
        **features,
    }


def analyze_location(lat, lng):
    prediction = predict_location(lat, lng)

    docs = []
    topics = []

    if app.state.retriever:
        try:
            docs, topics = app.state.retriever.retrieve(prediction)
        except Exception:
            logger.exception("Retriever failed")

    summary = None

    if app.state.generator:
        try:
            summary = app.state.generator.generate(lat, lng, prediction, docs)
        except Exception:
            logger.exception("Summary generation failed")

    return {
        "lat": lat,
        "lng": lng,
        "score": prediction["score"],
        "top_factors": prediction["top_factors"],
        "topics": topics,
        "n_docs": len(docs),
        "summary": summary,
        "ml_data": prediction,
    }


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def get_features(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
):
    try:
        return extract_features(lat, lng)

    except Exception:
        logger.exception("Failed extracting features")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/nearest")
def get_nearest(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    top_n: int = Query(3, ge=1, le=20),
):
    try:
        return {
            "competitors": nearest_records(app.state.cafes, lat, lng, top_n, ["name", "rating", "reviews_count", "lat", "lng"]),
            "owner_stores": nearest_records(app.state.owner, lat, lng, top_n),
        }

    except Exception:
        logger.exception("Failed loading nearest places")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/map-data")
def get_map_data(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    radius_m: float = Query(2000, ge=100, le=10000),
):
    try:
        poi_data = {}

        for category, df in app.state.poi.items():
            poi_data[category] = json.loads(nearby_records(df, lat, lng, radius_m).to_json(orient="records"))

        cafes = json.loads(nearby_records(app.state.cafes, lat, lng, radius_m, limit=200).to_json(orient="records"))

        owners = json.loads(app.state.owner.to_json(orient="records"))

        return {
            "cafes": cafes,
            "owner": owners,
            "poi": poi_data,
        }

    except Exception:
        logger.exception("Failed loading map data")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict")
def predict(body: Coord):
    try:
        if not in_jakarta(body.lat, body.lng):
            return {"valid": False,
                    "message": "Lokasi di luar cakupan (bukan DKI Jakarta)",
                    "score": None}
        return predict_location(body.lat, body.lng)

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze")
def analyze(body: Coord):
    try:
        if not in_jakarta(body.lat, body.lng):
            return {"valid": False,
                    "message": "Lokasi di luar cakupan (bukan DKI Jakarta)",
                    "score": None, "ml_data": None}
        return analyze_location(body.lat, body.lng)

    except Exception:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="Internal server error")
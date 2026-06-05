---
title: Coffee Shop Location AI
emoji: ☕
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# ☕ Coffee Shop Location Recommender — Jakarta

AI platform untuk rekomendasi lokasi pembukaan coffee shop di DKI Jakarta. Input koordinat (lat/long) → output skor kelayakan 0-100 + rekomendasi naratif dari AI.

## 📋 Overview

Owner F&B (coffee shop chain) butuh decision support untuk membuka cabang baru. Platform ini menggabungkan **ML scoring** + **AI narrative**:

- Hitung skor kelayakan lokasi 0-100 dari kondisi area
- Cek kanibalisasi dengan store existing owner
- Analisis kompetisi (cafe density, rating, review)
- Evaluasi sinyal demand (kantor, mall, transit, sekolah)
- Generate narasi rekomendasi GO / HOLD / NO-GO

Aplikasi dipecah jadi **dua bagian**: **API (FastAPI)** untuk inferensi model, dan **UI (Streamlit)** sebagai antarmuka. UI memanggil API. Data disimpan di **Supabase**.

---

## 🏗️ Architecture

```
Browser
   │
   ▼
Streamlit UI (app/streamlit_app.py)         ← antarmuka: peta + form + hasil
   │  HTTP (env API_URL)
   ▼
FastAPI (app/api.py)                         ← /predict /features /nearest /map-data /analyze
   │
   ├── DemandModel (XGBoost + SHAP)  ← models/xgb_demand.pkl
   ├── FeatureExtractor              ← src/features
   └── RAG: KnowledgeRetriever + SummaryGenerator (Groq LLM)
   │
   ▼
Supabase (Postgres)                          ← data via REST (src/data/supabase_io.py)
   tabel: raw_cafes, raw_poi_*, clean_cafes, clean_owner,
          clean_poi_*, owner_stores, kecamatan_ref, documents (RAG)
```

API & UI dibangun dari **satu image Docker** yang sama; `start.sh` memilih mode lewat env `APP_MODE` (`api` atau `ui`).

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy, scipy |
| Scraping | Google Places API, Overpass API (OSM) — via `requests` |
| ML | XGBoost, scikit-learn, SHAP, Optuna, MLflow |
| Database | Supabase (Postgres, diakses via REST/`requests`) |
| LLM | Groq Llama 3.3 70B |
| API | FastAPI + Uvicorn |
| UI | Streamlit + Folium |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions (ruff lint + build & push image ke GHCR) |
| Deploy | Hugging Face Spaces (Docker) |

---

## 📁 Folder Structure

```
project-root/
├── app/
│   ├── api.py                  # FastAPI: load model + data + endpoint inferensi
│   └── streamlit_app.py        # Streamlit UI (memanggil API)
│
├── src/
│   ├── data/
│   │   ├── data_loader.py      # DataLoader (baca dari Supabase)
│   │   ├── data_processing.py  # DataProcessor (clean + upload ke Supabase)
│   │   └── supabase_io.py      # baca/tulis Supabase via REST (requests)
│   ├── features/extractor.py   # FeatureExtractor
│   ├── models/
│   │   ├── model.py            # DemandModel (XGBoost + SHAP)
│   │   ├── trainer.py          # Trainer (MLflow + Optuna)
│   │   ├── evaluator.py        # Evaluator
│   │   ├── monitor.py          # ModelMonitor (drift)
│   │   └── retrain_trigger.py  # RetrainTrigger
│   ├── rag/
│   │   ├── retriever.py        # KnowledgeRetriever (Supabase `documents`)
│   │   └── generator.py        # SummaryGenerator (Groq LLM)
│   ├── scraping/
│   │   ├── scraping_gmaps.py   # Google Places -> Supabase
│   │   └── scraping_osm.py     # OSM Overpass -> Supabase
│   └── utils/geo.py            # haversine
│
├── scripts/seed_supabase.py    # one-time: CSV lokal -> Supabase
├── sql/supabase_schema.sql     # bikin tabel data di Supabase
├── test/                       # CLI: train_model, evaluate_model, monitor, retrain_trigger
├── notebook/                   # eksplorasi 01-06 (scraping, clean, FE, train, RAG)
├── models/                     # artifacts (xgb_demand.pkl via Git LFS, dll)
│
├── Dockerfile                  # 1 image; start.sh pilih API/UI lewat APP_MODE
├── start.sh                    # APP_MODE=api -> uvicorn ; APP_MODE=ui -> streamlit
├── docker-compose.yml          # lokal: service api + ui
├── .dockerignore / .gitignore / .gitattributes (LFS)
├── ruff.toml                   # config linter (CI)
├── .github/workflows/          # ci.yml (lint) + docker.yml (build & push GHCR)
├── requirements.txt
└── README.md
```

> `data/`, `mlruns/`, `logs/`, `venv/`, cache di-`.gitignore`. File model `.pkl`/`.png` disimpan via **Git LFS**.

---

## 🚀 Setup Lokal

### 1. Environment
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. `.env`
```
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJh...service_role_key
GROQ_API_KEY=gsk_...
GOOGLE_MAPS_API_KEY=AIza...
```

### 3. Supabase
- SQL Editor → jalankan `sql/supabase_schema.sql` (bikin tabel data).
- Tabel RAG `documents` (bikin table knowledge base)
- Isi data: `python3 scripts/seed_supabase.py` (dari CSV lokal) atau jalankan scraping.

---

## Running Docker

```bash
docker compose up --build
```
- UI  : http://localhost:8501
- API : http://localhost:8000/docs

---

## 🔌 API Endpoints

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/features?lat=&lng=` | Fitur lokasi (quick stats) |
| GET | `/nearest?lat=&lng=&top_n=` | Cafe & owner terdekat |
| GET | `/map-data?lat=&lng=&radius_m=` | Titik cafe/owner/POI buat peta |
| POST | `/predict` | ML score + faktor (body `{lat,lng}`) |
| POST | `/analyze` | Predict + RAG + ringkasan LLM |
| GET | `/docs` | Swagger UI |

---

## ☁️ Deployment (Hugging Face Spaces)

Dua Space dari repo yang sama (image Docker yang sama), dibedakan env `APP_MODE`:

| Space | APP_MODE | Variables/Secrets |
|---|---|---|
| API | `api` | secrets: `SUPABASE_URL`, `SUPABASE_KEY`, `GROQ_API_KEY` |
| UI | `ui` | `API_URL=https://<api-space>.hf.space` |

CI/CD: **GitHub Actions** — `ci.yml` (ruff lint + compile) tiap push/PR, `docker.yml` build & push image ke GitHub Container Registry tiap push `main`.

---

## 🧪 ML Model

**Fitur (24)**: kompetisi (`n_competitors_500m/2km`, rating, review), kanibalisasi (`nearest_owner_store_m`), POI density (office/mall/transit/school @500m & 2km), + 6 fitur turunan (`density_ratio`, `market_saturation`, `anchor_score`, dll).

**Target**: `rating × log(1 + reviews_count)` (proxy popularity dari ~2.000 cafe Jakarta).

**Model**: XGBoost regressor, tuning Optuna, tracking MLflow, explainability SHAP.

| Metric | Value |
|---|---|
| R² (CV 5-fold) | 0.25 ± 0.07 |
| MAE (CV) | 5.43 ± 0.18 |
| Spearman ρ | 0.55 |

---

## 🧠 RAG Pipeline

Rule-based retrieval: knowledge base di tabel Supabase `documents` dengan topic tags; topik dipilih dari output fitur ML (mis. `n_offices_500m`>30 → `cbd_strategy`). Generasi narasi pakai Groq Llama 3.3 70B (5 bagian: Ringkasan, Kekuatan, Risiko, Rekomendasi, Tindak Lanjut).

---

## 📈 Data Sources

| Source | Description | Count |
|---|---|---|
| Google Places API | Cafe Jakarta + rating + review | ~2.000 |
| OSM Overpass API | Office, mall, transit, school | ~15.000 |
| Owner Stores | Store existing owner | 18 |

Semua data disimpan di Supabase (1 tabel per dataset).

---

## 🔮 Future Work

- Hybrid RAG (rule-based + vector embedding)
- Tambah data populasi BPS & kampus PDDIKTI
- Always-on deploy (Render/VPS) untuk hilangkan cold-start
- Export PDF report
```

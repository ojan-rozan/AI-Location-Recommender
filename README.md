---
title: Coffee Shop Location AI
emoji: ☕
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# ☕ Coffee Shop Location Recommender — Jakarta

AI platform untuk rekomendasi lokasi pembukaan coffee shop di DKI Jakarta. Input koordinat (lat/long) → output skor kelayakan 0-100 + AI narrative recommendation.

**Capstone Project — Dibimbing**

---

## 📋 Overview

Owner F&B (coffee shop chain) butuh decision support untuk membuka cabang baru. Platform ini kombinasi **ML scoring** + **AI narrative** untuk:

- ✅ Hitung skor kelayakan lokasi 0-100 berdasarkan kondisi area
- ✅ Cek kanibalisasi dengan store existing owner
- ✅ Analisis kompetisi (cafe density, rating, review)
- ✅ Evaluasi sinyal demand (kantor, mall, transit, sekolah)
- ✅ Generate AI narrative dengan rekomendasi GO / HOLD / NO-GO

Aplikasi dipecah jadi **dua service**: UI (Streamlit) dan API (FastAPI). UI tipis dan memanggil API; semua logika inferensi ada di API. Data disimpan di **Supabase**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│            Browser                          │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│   STREAMLIT UI  (app/streamlit_app.py:8501)  │
│   Map folium + quick stats + AI summary      │
│   → manggil API via HTTP (requests)          │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│   FASTAPI  (app/api.py:8000)                 │
│   /features /nearest /map-data /predict      │
│   /analyze /health /docs                     │
│        │                    │                │
│        ▼                    ▼                │
│  ┌───────────┐      ┌────────────────┐       │
│  │ ML SCORING│      │  RAG PIPELINE  │       │
│  │ Features  │      │  Supabase docs │       │
│  │ XGBoost   │      │  (rule-based)  │       │
│  │ SHAP      │      │  Groq LLM      │       │
│  └─────┬─────┘      └───────┬────────┘       │
└────────┼────────────────────┼────────────────┘
         ▼                    ▼
┌─────────────────────────────────────────────┐
│   DATA LAYER — SUPABASE                       │
│   • raw_cafes / raw_poi_* (hasil scraping)    │
│   • clean_cafes / clean_owner / clean_poi_*   │
│   • owner_stores / kecamatan_ref              │
│   • documents (RAG knowledge base)            │
│   + model file lokal (models/xgb_demand.pkl)  │
└─────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy, geopandas |
| Scraping | Google Places API, Overpass API (OSM) |
| ML | XGBoost, scikit-learn, SHAP, Optuna |
| Experiment Tracking | MLflow |
| Database / Knowledge | Supabase (Postgres + JSONB) |
| LLM | Groq Llama 3.3 70B |
| API | FastAPI + Uvicorn |
| UI | Streamlit + Folium |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions (ruff lint + build & push image ke GHCR) |

---

## 📁 Folder Structure

```
project-root/
├── app/
│   ├── api.py                  # FastAPI — logika inferensi + endpoint
│   └── streamlit_app.py        # Streamlit UI (manggil API)
│
├── src/                        # importable modules
│   ├── data/
│   │   ├── data_loader.py      # DataLoader — baca dari Supabase
│   │   ├── data_processing.py  # DataProcessor — clean + upload ke Supabase
│   │   └── supabase_io.py      # helper baca/tulis DataFrame ↔ Supabase
│   ├── features/extractor.py   # FeatureExtractor
│   ├── models/
│   │   ├── model.py            # DemandModel (XGBoost wrapper)
│   │   ├── trainer.py          # Trainer (MLflow + Optuna)
│   │   ├── evaluator.py        # Evaluator
│   │   ├── monitor.py          # ModelMonitor (drift)
│   │   └── retrain_trigger.py  # RetrainTrigger
│   ├── rag/
│   │   ├── retriever.py        # KnowledgeRetriever (Supabase `documents`)
│   │   └── generator.py        # SummaryGenerator (Groq LLM)
│   ├── scraping/
│   │   ├── scraping_gmaps.py   # Google Places → Supabase
│   │   └── scraping_osm.py     # OSM POI → Supabase
│   └── utils/geo.py            # haversine
│
├── scripts/
│   └── seed_supabase.py        # one-time: CSV lokal → Supabase
├── sql/
│   └── supabase_schema.sql     # bikin tabel data di Supabase
│
├── test/                       # CLI entry-point (bukan unit test)
│   ├── train_model.py          # full training pipeline
│   ├── evaluate_model.py
│   ├── monitor.py
│   └── retrain_trigger.py
│
├── notebook/                   # exploratory notebooks (01–06)
├── models/                     # trained artifacts (xgb_demand.pkl, dll)
│
├── Dockerfile                  # image (default CMD = API)
├── docker-compose.yml          # service: api + ui
├── .dockerignore
├── ruff.toml                   # config linter (CI)
├── .github/workflows/
│   ├── ci.yml                  # lint + compile check
│   └── docker.yml              # build & push image ke GHCR
├── requirements.txt
├── .env / .env.example
└── README.md
```

> Catatan: `data/`, `mlruns/`, `logs/`, `venv/` di-`.gitignore` (data sumbernya Supabase). `models/` ikut commit karena dibutuhkan saat runtime.

---

## 🚀 Setup

### 1. Clone + virtual environment

```bash
git clone <repo-url>
cd "Project Akhir - DIbimbing"

python3 -m venv venv
source venv/bin/activate        # Linux/Mac
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` → `.env`, isi credentials:

```
GOOGLE_MAPS_API_KEY=AIzaSy...        # buat scraping (opsional)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJh...service_role_key  # service_role buat tulis data
GEMINI_API_KEY=...                    # opsional
GROQ_API_KEY=gsk_...                  # buat LLM
```

### 3. Setup Supabase

**a. Tabel data** — buka Supabase SQL Editor, jalanin isi `sql/supabase_schema.sql` (bikin 13 tabel: `raw_cafes`, `clean_cafes`, `raw_poi_*`, dll).

**b. Tabel RAG knowledge base** (`documents`):

```sql
create table if not exists documents (
    id text primary key,
    content text not null,
    metadata jsonb,
    created_at timestamptz default now()
);
create index if not exists documents_metadata_idx on documents using gin(metadata);
```

### 4. Isi data ke Supabase

Kalau sudah punya CSV lokal (hasil scraping sebelumnya), seed sekali:

```bash
python3 scripts/seed_supabase.py
```

Atau scrape dari awal (otomatis upload ke Supabase):

```bash
python3 src/scraping/scraping_gmaps.py   # cafe Google Maps
python3 src/scraping/scraping_osm.py     # POI OSM
```

### 5. Get API keys

- **Supabase**: https://supabase.com → Settings → API
- **Groq**: https://console.groq.com/keys (free tier)
- **Google Places** (opsional): https://console.cloud.google.com

---

## ▶️ Menjalankan

### Pakai Docker (rekomendasi) — API + UI sekaligus

```bash
docker compose up --build
```

- **UI Streamlit** → http://localhost:8501
- **API + Swagger docs** → http://localhost:8000/docs

### Tanpa Docker (2 terminal)

```bash
# terminal 1 — API
uvicorn app.api:app --port 8000

# terminal 2 — UI
streamlit run app/streamlit_app.py
```

UI baca alamat API dari env `API_URL` (default `http://localhost:8000`).

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
| GET | `/docs` | Swagger UI interaktif |

Contoh:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"lat":-6.2297,"lng":106.8195}'
```

---

## 🧪 ML Model

### Feature Engineering

24 features (18 base + 6 derivative):

**Base (18)**: kompetisi (`n_competitors_500m`, `n_competitors_2km`, rating avg, total review, max review), cannibalization (`nearest_owner_store_m`), POI density (office, mall, transit, school — radius 500m + 2km).

**Derivative (6)**: `density_ratio_500m_2km`, `avg_reviews_per_cafe_2km`, `market_saturation`, `office_transit_combo`, `has_strong_competitor`, `anchor_score`.

### Target

```python
target = rating × log(1 + reviews_count)
```

Proxy popularity dari cafe Google Maps (~2.000 cafe Jakarta).

### Model & Performance

XGBoost regressor, tuning Optuna, tracking MLflow.

| Metric | Value |
|---|---|
| R² (CV 5-fold) | 0.25 ± 0.07 |
| MAE (CV) | 5.43 ± 0.18 |
| Spearman ρ | 0.55 |

**Primary metric**: Spearman ρ — ranking quality untuk use case rekomendasi lokasi.

---

## 🧠 RAG Pipeline

**Rule-based retrieval** (bukan vector embedding). Knowledge base curated di tabel Supabase `documents` dengan topic tags; retrieval pakai metadata filter berdasarkan ML feature output.

| Feature condition | Topic |
|---|---|
| `n_offices_500m` > 30 | `cbd_strategy` |
| `n_offices_500m` < 5 | `residential_strategy` |
| `n_competitors_500m` > 15 | `high_competition` |
| `n_competitors_500m` < 3 | `low_competition_opportunity` |
| `n_transit_500m` > 3 | `transit_advantage` |
| `n_malls_2km` > 0 | `mall_anchor` |
| `nearest_owner_store_m` < 1000 | `cannibalization` |

**Generation**: Groq Llama 3.3 70B → narrative 5 bagian (Ringkasan, Kekuatan, Risiko, Rekomendasi, Tindak Lanjut).

---

## 🔄 Training & MLOps

### Full pipeline (load Supabase → clean → train → upload clean)

```bash
python3 test/train_model.py
```

### Steps lain

```bash
python3 test/evaluate_model.py     # evaluate model
python3 test/monitor.py            # monitor performa + drift
python3 test/retrain_trigger.py    # cek auto-retrain
mlflow ui                          # http://localhost:5000
```

---

## 🐳 Docker & CI/CD

**Dockerfile** — image tunggal berisi API + UI. CMD default = API (uvicorn); service UI nimpa command jadi streamlit di `docker-compose.yml`.

**GitHub Actions:**

- `ci.yml` — tiap push & PR: ruff lint + compile check.
- `docker.yml` — tiap push ke `main`: build image + push ke **GHCR** (`ghcr.io/<user>/<repo>`). Pas PR cuma build (verifikasi).

Tarik & jalanin image hasil build:

```bash
docker pull ghcr.io/<user>/<repo>:latest
docker run -p 8000:8000 --env-file .env ghcr.io/<user>/<repo>:latest
```

> Image GHCR default private — ubah ke public di GitHub → Packages → Settings kalau mau di-pull bebas.

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

- [ ] Upgrade ke hybrid RAG (rule-based + vector embedding)
- [ ] Add data populasi BPS per kelurahan
- [ ] Integrate data kampus PDDIKTI
- [ ] Time-series feature (cafe age, growth trend)
- [ ] A/B test comparison view (2 lokasi side-by-side)
- [ ] Auto-deploy ke hosting (Render/Railway/Fly.io)
- [ ] Export PDF report
```

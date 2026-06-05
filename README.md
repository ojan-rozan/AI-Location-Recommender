---
title: Coffee Shop Location AI
emoji: ☕
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Coffee Shop Location Recommender

## Problem

Menentukan lokasi cabang baru coffee shop sering dilakukan berdasarkan intuisi.
Project ini mencoba membantu proses tersebut menggunakan data kompetitor,
POI, dan machine learning.

## Data

- Google Places API (~2.000 cafe Jakarta)
- OpenStreetMap (~15.000 POI)
- Data owner store (18 lokasi)

## Features

- Jumlah kompetitor 500m dan 2km
- Rating dan review kompetitor
- Jarak ke owner store terdekat
- Office, mall, transit, school density
- Feature engineering (market saturation, anchor score, dll)

## Model

XGBoost Regressor

Target:
rating × log(1 + review_count)

Performance:

| Metric | Value |
|---------|---------|
| CV R² | 0.25 ± 0.07 |
| CV MAE | 5.43 ± 0.18 |
| Spearman | 0.55 |

## Application

- FastAPI backend
- Streamlit frontend
- Supabase storage
- Groq LLM untuk ringkasan lokasi

## How to Run

docker compose up --build

UI:
localhost:8501

API:
localhost:8000/docs

## Future Improvement

- Population data integration
- Embedding-based RAG
- PDF report export
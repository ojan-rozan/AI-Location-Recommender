# Base image
FROM python:3.11-slim

WORKDIR /app

# Dependency sistem buat geopandas/osmnx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lib python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy kode 
COPY . .

EXPOSE 8000 8501

# Health check — Streamlit (default port 8501 / $PORT)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

# CMD default = WEBSITE Streamlit (listen di $PORT, default 8501).
# Service "api" di docker-compose nimpa command jadi uvicorn.
CMD streamlit run app/streamlit_app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false

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

# Health check default buat service API (uvicorn)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# CMD default = API. Service UI nimpa command-nya di docker-compose.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

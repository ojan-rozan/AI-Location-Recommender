# Base image
FROM python:3.11-slim

WORKDIR /app

# Install lib python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy kode
COPY . .

# Port: HF default 7860; lokal bisa di-set via env PORT
EXPOSE 7860 8000 8501

# start.sh milih API atau UI berdasarkan env APP_MODE (default: api)
CMD ["bash", "start.sh"]

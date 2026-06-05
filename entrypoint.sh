set -e
PORT="${PORT:-7860}"

if [ "$APP_MODE" = "api" ]; then
    echo "Starting FastAPI on :$PORT"
    exec uvicorn app.api:app --host 0.0.0.0 --port "$PORT"
else
    echo "Starting Streamlit on :$PORT"
    exec streamlit run app/streamlit_app.py \
        --server.port "$PORT" \
        --server.address 0.0.0.0 \
        --server.enableCORS false \
        --server.enableXsrfProtection false
fi

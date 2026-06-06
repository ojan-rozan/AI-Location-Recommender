set -e

PORT="${PORT:-7860}"

if [ "$APP_MODE" = "ui" ]; then
    exec streamlit run app/streamlit_app.py \
        --server.port "$PORT" \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        --server.enableWebsocketCompression false \
        --server.fileWatcherType none \
        --server.runOnSave false \
        --browser.gatherUsageStats false
else
    exec uvicorn app.api:app --host 0.0.0.0 --port "$PORT"
fi

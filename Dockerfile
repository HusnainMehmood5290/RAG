FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models

# Non-root runtime user (drop privileges; chown only the dirs we own).
RUN groupadd --system app && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .

# Pre-writable store dirs (mount volumes in production for persistence)
RUN mkdir -p store/local_store store/vector_store ingestion/raw_data ingestion/processed \
    && chown -R app:app /app/store /app/ingestion /models

USER app

EXPOSE 8501
# Security-relevant server flags (auth is handled out of scope for now):
#   headless                 - no browser-launch attempts / extra endpoints
#   disableStaticCaching     - stale cached assets can pin old vulnerable JS
#   enableXsrfProtection     - defense-in-depth against cross-site POSTs
# NOTE: expose the app only behind a reverse proxy / VPN until authentication
# is added; do not bind this container directly to public networks.
CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.disableStaticCaching=true", \
     "--server.enableXsrfProtection=true"]

# syntax=docker/dockerfile:1

# ── Stage 1 : build du frontend ────────────────────────────────────────────
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ── Stage 2 : runtime Python ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Dépendances système :
# - libreoffice (writer/calc/impress) : conversion des Office legacy .doc/.xls/.ppt
# - ocrmypdf + tesseract (fra+eng) : OCR optionnel des PDF scannés
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer libreoffice-calc libreoffice-impress \
        ocrmypdf tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch CPU-only d'abord (évite les paquets CUDA), puis le projet.
COPY pyproject.toml README.md ./
COPY rag_builder/ ./rag_builder/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[gemini,api,mcp,mistral,anthropic]"

# Front buildé (servi en statique par FastAPI).
COPY --from=web /web/dist ./web/dist
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/download_models.py ./download_models.py
RUN chmod +x /entrypoint.sh

# Réglages runtime : Qdrant serveur, modèles en volume, hors-ligne HF, storage en volume.
ENV QDRANT_MODE=server \
    QDRANT_URL=http://qdrant:6333 \
    MODELS_CACHE_DIR=/models \
    HF_OFFLINE=true \
    STORAGE_DIR=/data/storage \
    DATA_DIR=/data/incoming \
    COOKIE_SECURE=true

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]

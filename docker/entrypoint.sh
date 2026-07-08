#!/usr/bin/env bash
# Entrypoint de l'image API : crée un admin initial si demandé, puis lance la commande.
set -euo pipefail

# Bootstrap du premier administrateur (idempotent : ignore si déjà présent).
if [[ -n "${RAGB_ADMIN_USER:-}" && -n "${RAGB_ADMIN_PASSWORD:-}" ]]; then
  python -m rag_builder create-user "$RAGB_ADMIN_USER" \
      --password "$RAGB_ADMIN_PASSWORD" --admin || true
fi

if [[ "${1:-serve}" == "serve" ]]; then
  exec python -m rag_builder serve --host 0.0.0.0 --port 8000
fi

exec python -m rag_builder "$@"

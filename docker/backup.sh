#!/usr/bin/env bash
# Sauvegarde RAG Builder : base SQLite + snapshot Qdrant. Prêt pour cron.
#
#   crontab -e :
#   0 2 * * *  /chemin/rag_poc/docker/backup.sh >> /var/log/ragb-backup.log 2>&1
#
# Variables (avec valeurs par défaut) :
#   BACKUP_DIR   répertoire de destination            (./backups)
#   COMPOSE      commande compose                      (docker compose)
#   QDRANT_URL   URL Qdrant depuis l'hôte              (http://localhost:6333) — via le réseau compose sinon
#   KEEP         nb de sauvegardes à conserver         (14)
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./backups}"
COMPOSE="${COMPOSE:-docker compose}"
KEEP="${KEEP:-14}"
TS="$(date +%Y%m%d_%H%M%S)"
DEST="$BACKUP_DIR/$TS"
mkdir -p "$DEST"

echo "[backup $TS] → $DEST"

# 1) SQLite : copie cohérente via `.backup` (gère le WAL).
$COMPOSE exec -T api sh -c 'sqlite3 /data/storage/app.db ".backup /tmp/app.db" 2>/dev/null || cp /data/storage/app.db /tmp/app.db'
$COMPOSE cp api:/tmp/app.db "$DEST/app.db"

# 2) Qdrant : snapshot complet via l'API, puis récupération du fichier.
$COMPOSE exec -T qdrant sh -c 'curl -s -X POST http://localhost:6333/snapshots >/dev/null'
# Copie l'intégralité du stockage Qdrant (inclut les snapshots générés).
$COMPOSE cp qdrant:/qdrant/storage "$DEST/qdrant_storage"

# 3) Images extraites (référencées par les réponses).
$COMPOSE cp api:/data/storage/images "$DEST/images" 2>/dev/null || true

# 4) Archive + rotation.
tar -czf "$DEST.tar.gz" -C "$BACKUP_DIR" "$TS" && rm -rf "$DEST"
echo "[backup $TS] archive : $DEST.tar.gz"

ls -1dt "$BACKUP_DIR"/*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
echo "[backup $TS] terminé (rétention : $KEEP)"

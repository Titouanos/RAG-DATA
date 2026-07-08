# Déploiement

Deux modes : **dev** (sans Docker, Qdrant embarqué) et **production** (Docker Compose :
api+worker, Qdrant serveur, reverse proxy TLS Caddy, front buildé, modèles pré-embarqués).

---

## Mode dev (sans Docker)

Qdrant tourne en embarqué (`QDRANT_MODE=local`), aucun service à lancer.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[gemini,api,mcp,dev]"

# Modèles locaux (une fois) — ~4,3 Go dans storage/models_cache
export HF_HOME="$PWD/storage/models_cache"
python docker/download_models.py            # ou snapshot_download (cf. README)

cp .env.example .env                        # renseigner la clé du provider LLM
python -m rag_builder create-user admin --admin
python -m rag_builder serve --port 8000     # API + worker + front (si web/dist buildé)

# Frontend en dev (hot reload) : autre terminal
cd web && npm install && npm run dev        # http://localhost:5173 (proxy API → :8000)
```

Éval de non-régression : `python eval/run_eval.py` (API arrêtée en mode local).

---

## Mode production (Docker Compose)

Cible : serveur Linux interne, CPU (pas de GPU). Au runtime, **le seul flux sortant est vers
l'API du provider LLM** — les modèles locaux (bge-m3, reranker) sont pré-téléchargés une
seule fois par le service `models-init`, puis `HF_OFFLINE=true` interdit tout accès HF.

```bash
cp .env.docker.example .env
# éditer .env : MISTRAL_API_KEY (ou autre provider), RAGB_ADMIN_PASSWORD,
#               CADDY_DOMAIN (domaine interne), PUBLIC_URL

docker compose up -d --build
```

Séquence au premier démarrage :
1. `models-init` télécharge bge-m3 + reranker dans le volume `models` (accès HF unique), puis s'arrête ;
2. `qdrant` démarre (volume `qdrant_storage`) ;
3. `api` démarre (worker inclus), crée l'admin depuis `RAGB_ADMIN_*`, sert l'API + le front ;
4. `caddy` publie en HTTPS sur `CADDY_DOMAIN` (certificat interne auto-signé par défaut ;
   pour un domaine public, remplacer `tls internal` par `tls admin@…` dans `docker/Caddyfile`).

Accès : `https://<CADDY_DOMAIN>`. Vérifier : `docker compose ps`, `docker compose logs -f api`.

> Après la première création de l'admin, retirer `RAGB_ADMIN_PASSWORD` du `.env`.
> Changer d'embedding pour une collection existante impose une réindexation (embedding figé).

### Sauvegarde (cron)

`docker/backup.sh` sauvegarde la base SQLite (copie cohérente via `.backup`), le stockage
Qdrant (snapshot) et les images extraites, en archive horodatée avec rotation.

```bash
0 2 * * *  /chemin/rag_poc/docker/backup.sh >> /var/log/ragb-backup.log 2>&1
```

### Mise à jour

```bash
git pull && docker compose up -d --build      # rebuild api (front inclus) ; models-init no-op
```

---

## Serveur MCP

Le RAG est aussi exposé en MCP (`rag_query(collection, question)`, `list_collections`,
`rag_stats`), client du même cœur.

```bash
python -m rag_builder mcp                                  # stdio (Claude Desktop, mcpo…)
python -m rag_builder mcp --transport streamable-http --port 8100 \
       --image-base-url https://<CADDY_DOMAIN>             # HTTP + réécriture des images
```

En mode Qdrant local, lancer le MCP **seul** (accès exclusif à `storage/qdrant`) ; en mode
serveur (Docker), il peut cohabiter avec l'API.

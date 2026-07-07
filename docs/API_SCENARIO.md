# Scénario API de bout en bout (DoD Phase 2)

Parcours complet en `curl` : créer un compte → se connecter → créer une collection →
uploader un PDF → suivre le job → poser une question streamée (SSE) avec sources →
supprimer le document → vérifier que ses chunks ont disparu.

## Prérequis

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[gemini,api,dev]"
export HF_HOME="$PWD/storage/models_cache"     # cache des modèles locaux
# (facultatif) génération LLM : renseigner la clé, sinon l'event `sources` arrive mais
# la génération renvoie un event `error`.
export GEMINI_API_KEY=...

# 1) Créer le premier administrateur (crée aussi storage/app.db)
python -m rag_builder create-user admin --admin        # demande le mot de passe

# 2) Lancer l'API (worker d'ingestion démarré automatiquement)
python -m rag_builder serve --host 127.0.0.1 --port 8000
```

## Parcours

```bash
BASE=http://127.0.0.1:8000
COOKIES=/tmp/ragb_cookies.txt

# Santé
curl -s $BASE/health
# → {"status":"ok","embedder":"local_bge_m3"}

# Connexion (dépose un cookie de session httpOnly)
curl -s -c $COOKIES -X POST $BASE/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"<mot de passe>"}'
# → {"id":1,"username":"admin","role":"admin"}

# Créer une collection (réservé aux admins par défaut)
curl -s -b $COOKIES -X POST $BASE/collections \
  -H 'Content-Type: application/json' \
  -d '{"name":"kb","description":"Base de démo"}'
# → {"name":"kb","embedding_model":"BAAI/bge-m3","dense_dim":1024,"rerank_enabled":true,...}

# Uploader un PDF (multipart ; plusieurs -F files=@… possibles) → crée un job
curl -s -b $COOKIES -X POST $BASE/collections/kb/documents \
  -F "files=@teamcenter.pdf"
# → {"jobs":[{"job_id":1,"source_name":"teamcenter.pdf","doc_id":"doc_37ecc686..."}]}

# Suivre le job (statut + progression par étape)
curl -s -b $COOKIES $BASE/jobs/1
# → {"status":"running","stage":"embedding","progress_current":32,"progress_total":128,...}
# … puis :
# → {"status":"succeeded","stage":"done","message":"new",...}

# Lister les documents (statut, chunks)
curl -s -b $COOKIES $BASE/collections/kb/documents
# → [{"doc_id":"doc_37ecc686...","source_name":"teamcenter.pdf","status":"indexed","n_chunks":1,...}]

# Poser une question — réponse en SSE (event sources → token* → done)
curl -sN -b $COOKIES -X POST $BASE/collections/kb/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Comment reinitialiser le mot de passe FSC ?"}'
# event: sources
# data: {"sources": [{"n": 1, "source_name": "teamcenter.pdf", "page_or_section": "teamcenter > Page 1", "score": 1.0, "chunk_id": "...", "excerpt": "..."}]}
#
# event: token
# data: {"t": "Pour "}
# event: token
# data: {"t": "réinitialiser "}
# …
# event: done
# data: {"timings": {"embed_ms": 305.1, "search_ms": 4.8, "rerank_ms": 640.2, "total_ms": 950.1}}

# Feedback (👍/👎) sur la dernière réponse
curl -s -b $COOKIES -X POST $BASE/collections/kb/feedback \
  -H 'Content-Type: application/json' \
  -d '{"question":"...","rating":"up","chunk_ids":["..."]}'

# Supprimer le document → ses chunks disparaissent immédiatement (dense + sparse)
curl -s -b $COOKIES -X DELETE $BASE/collections/kb/documents/doc_37ecc686...
# → {"status":"deleted","doc_id":"doc_37ecc686...","chunks_removed":1}

# Vérifier : la même question ne renvoie plus ce document
curl -sN -b $COOKIES -X POST $BASE/collections/kb/query \
  -H 'Content-Type: application/json' -d '{"question":"reinitialiser FSC"}'
# event: sources
# data: {"sources": []}
```

## Notes

- **Progression** : le worker met à jour `stage` (`parsing` → `embedding` → `indexing` →
  `done`) et `progress_current/total` (ex. `embedding 32/128`), lisibles via `GET /jobs/{id}`.
- **Robustesse** : un PDF corrompu/protégé → job `failed` avec message clair, jamais de crash
  du worker ; au redémarrage, les jobs restés `running` repassent `pending` ; hash déjà
  présent → job `skipped`. Taille d'upload max configurable (`MAX_UPLOAD_MB`, défaut 100).
- **Sécurité** : sessions cookie httpOnly (argon2) ; création/suppression de collections
  réservée aux admins par défaut (`COLLECTIONS_ADMIN_ONLY`). En prod derrière TLS :
  `COOKIE_SECURE=true`.
- **OCR** : les PDF dont >30 % des pages sont sans couche texte sont marqués
  `scanned_suspect=true` sur le document (l'UI suggérera l'OCR — implémenté en Phase 4).

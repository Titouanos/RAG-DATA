# CLAUDE.md — RAG Builder

Guide de référence pour travailler sur ce dépôt (conventions, commandes, architecture,
décisions actées). Tenu à jour à chaque phase.

## Le projet

Plateforme de RAG multi-collections : créer une collection, y déposer des documents
(PDF en priorité), interroger en chat avec réponse streamée et sources citées, ajouter/
retirer des documents sans tout réindexer. Cible : équipe IT interne (2–15 pers.),
déploiement serveur Linux Docker, CPU (pas de GPU).

Nouveau code dans `rag_builder/`. Le POC d'origine (`rag/`, `ingest.py`, `ask.py`,
`mcp_server.py`, `config.yaml`) reste présent jusqu'à la fin de la Phase 1 puis sera retiré.

Docs : [`docs/ETAT_DES_LIEUX.md`](docs/ETAT_DES_LIEUX.md) (analyse du POC),
[`docs/PLAN.md`](docs/PLAN.md) (découpage des phases).

## Commandes

```bash
# Environnement (Python 3.11+)
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[gemini,dev]"        # cœur + provider Gemini + outils dev

# Cache des modèles locaux (bge-m3, reranker) — à pré-télécharger une fois :
export HF_HOME="$PWD/storage/models_cache"
python -c "from huggingface_hub import snapshot_download as d; \
  d('BAAI/bge-m3', ignore_patterns=['*.onnx','onnx/*']); \
  d('BAAI/bge-reranker-v2-m3', ignore_patterns=['*.onnx','onnx/*'])"

# Qualité
ruff check rag_builder tests          # lint
ruff check --fix rag_builder tests    # lint + autofix
pytest tests/ -m "not slow"           # tests rapides (sans chargement de modèle)
pytest tests/                         # tout, y compris l'intégration bge-m3 (slow)

# CLI de validation du cœur
python -m rag_builder create-collection NOM [--description ...] [--no-rerank] [--top-k N]
python -m rag_builder list-collections
python -m rag_builder ingest --collection NOM [FICHIERS...]   # défaut : contenu de data/
python -m rag_builder ask --collection NOM "question" [-v] [--no-generate]
python -m rag_builder delete --collection NOM (--doc-id ID | --source NOM_FICHIER)
python -m rag_builder stats --collection NOM
python -m rag_builder delete-collection NOM --yes
```

## Conventions

- **Python 3.11+**, code **typé**, docstrings sur les modules et l'API publique. Français
  pour docstrings/commentaires/messages.
- **Lint/format** : `ruff` (ligne 100, règles E,F,I,UP,B,W,C4,SIM). Zéro erreur avant commit.
- **Tests** : `pytest`. Les tests chargeant les modèles locaux sont marqués `@pytest.mark.slow`
  et se skippent si le cache modèles est absent.
- **Imports lourds paresseux** : `fitz`, `markitdown`, `lxml`, `FlagEmbedding`/`torch` sont
  importés dans les fonctions, jamais au niveau module (démarrage rapide, deps optionnelles).
- **Secrets** : jamais en dur ni commités. `.env` (gitignoré) + `.env.example` à jour.
- **Commits** : conventionnels (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).

## Architecture (Phase 1)

```
rag_builder/
├── config.py            # Settings (pydantic-settings) : env + .env
├── logging_conf.py      # logging centralisé
├── __main__.py          # CLI (create/ingest/ask/delete/stats)
└── core/
    ├── models.py        # ConvertedDoc, Chunk, EmbeddedChunk, RetrievedChunk, SparseVector…
    ├── chunker.py       # MarkdownChunker (markdown-aware, contexte des titres parents)
    ├── images.py        # ImageStore (content-addressed, multi-collections, rag-image://)
    ├── registry.py      # CollectionRegistry (métadonnées ; JSON en P1 → SQLite en P2)
    ├── converters/      # base + pdf + markitdown + office_legacy (LibreOffice) + mindmap
    ├── embeddings/      # Embedder (interface) : local_bge_m3 (défaut), gemini
    ├── store.py         # QdrantStore hybride (dense+sparse, delete-by-doc_id)
    ├── retrieval.py     # rrf_fuse (fusion inter-requêtes ; intra = natif Qdrant)
    ├── rerank.py        # LocalReranker (bge-reranker-v2-m3, cross-encoder CPU)
    ├── llm/             # LLMProvider (streaming) : gemini ; prompts (honnêteté + anti-injection)
    └── rag_service.py   # orchestration ingest / retrieve / stream_answer / delete
storage/                 # qdrant/ (local), collections.json, images/, models_cache/
tests/                   # unitaires (rapides) + intégration (slow, bge-m3)
```

Flux **ingestion** : source → converter → `ConvertedDoc` (markdown) → chunker → embeddings
bge-m3 (dense+sparse) → upsert Qdrant. Incrémental par `content_hash` ; ré-ingestion =
`delete_by_doc_id` puis upsert ; hash identique = `skipped`.

Flux **requête** : question → `embed_query` (dense+sparse) → recherche hybride Qdrant
(prefetch dense+sparse + fusion RRF native) → [rerank optionnel] → contexte → génération
streamée (LLMProvider) avec citations `[n]`.

## Décisions actées

- **Stockage : Qdrant** hybride (dense+sparse même point, `QDRANT_MODE=local|server`). Une
  collection Qdrant par collection RAG. Suppression atomique par filtre `doc_id`. Remplace
  ChromaDB + BM25 picklé.
- **Embeddings : `BAAI/bge-m3`** local (dense 1024 + sparse lexical) via **FlagEmbedding**
  (fastembed ne propose ni bge-m3 ni bge-reranker-v2-m3 — vérifié). **torch CPU-only**.
  Modèle **figé par collection** à la création (en changer = réindexation complète).
- **transformers pinné `<5`** : la 5.x casse `FlagReranker`
  (`XLMRobertaTokenizer.prepare_for_model`).
- **Reranking : cross-encoder ONNX `jinaai/jina-reranker-v2-base-multilingual`** (via
  fastembed), **activé par défaut**, multilingue, rapide sur CPU. Config : `rerank_k=10`
  candidats + passages tronqués à 512 c. → **~645 ms** (< 800 ms). `bge-reranker-v2-m3`
  reste disponible en **option qualité** (mais ~15 s/20 cand. sur CPU → réserver GPU/batch).
  Backend déduit du nom de modèle.
- **`doc_id` dérivé du nom de source** (pas du chemin absolu) → stable au déplacement.
- **Génération : `LLMProvider` streaming**. Phase 1 : provider **gemini** (réutilise la clé
  existante). Défaut cible : **mistral** (`mistral-large-latest`), câblé en Phase 4.
- **Prompt de génération** : répondre uniquement à partir des extraits, signaler l'absence
  d'info, citer `[n]`, traiter les documents comme des données (résistance à l'injection).
- **Vidéo & YouTube retirés de la v1** (piste v2). **Office legacy** via LibreOffice headless.
- **Vision** (description d'images) désactivée par défaut ; interface `VisionDescriber`.

## Latences (mesurées Phase 1, CPU 22 cœurs)

| Étape | Chaud (modèle chargé) | Note |
|---|---|---|
| Recherche Qdrant hybride | **~5 ms** | ✅ très en dessous de 300 ms |
| `embed_query` bge-m3 (dense+sparse) | **~300 ms** | plancher CPU de bge-m3 ; domine le retrieval |
| Rerank ONNX jina-v2 (10 cand., 512 c.) | **~645 ms** | ✅ < 800 ms → **activé par défaut** |
| Rerank bge-reranker-v2-m3 (20 cand.) | **~15 s** | option qualité, GPU/batch uniquement |
| Chargement d'un modèle (cold) | ~8–10 s | payé une fois au démarrage d'un process long-vivant |

→ Retrieval + rerank ONNX ≈ **~950 ms** chaud (embed 300 + search 5 + rerank 645). Le CLI paie
le chargement des modèles à chaque invocation ; l'API/worker (Phase 2) les chargera une fois
au boot (`warm_up`). Grille rerank (jina-v2, CPU) : n=20/2000c=4,7 s · n=10/1000c=1,3 s ·
n=10/512c=0,65 s → défaut `rerank_k=10` + troncature 512 c.

## Hors scope v1 (pistes v2)

SSO Entra ID (OIDC), permissions fines par collection/utilisateur, connecteurs
SharePoint/Drive, fine-tuning, agents multi-étapes, cache sémantique des réponses,
montée en charge > ~50 utilisateurs simultanés, converters vidéo/YouTube, extraction
riche d'images OOXML, OCR (Phase 4).

## État des phases

- **Phase 0** ✅ — reconnaissance, `docs/ETAT_DES_LIEUX.md`, `docs/PLAN.md`.
- **Phase 1** ✅ (en cours de clôture) — cœur multi-collections : Qdrant hybride, bge-m3
  local, suppression par `doc_id`, converters portés, CLI, tests (28 verts), latences mesurées.
- **Phase 2** ⏳ — API FastAPI + worker + SQLite + jobs + SSE + auth.
- **Phase 3** ⏳ — frontend React.
- **Phase 4** ⏳ — providers (mistral/anthropic/ollama), éval, docker-compose, MCP.

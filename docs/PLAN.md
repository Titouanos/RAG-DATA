# Plan de migration — POC `rag_poc` → RAG Builder

> Découpage détaillé des Phases 1→4 en tâches. Chaque phase se termine par :
> **tests verts + `ruff` OK + commits conventionnels + compte-rendu** (fait / comment
> tester à la main / reste à faire). Voir `docs/ETAT_DES_LIEUX.md` pour la justification
> module par module. Les **[VALIDER]** sont les points où je demanderai ton feu vert.

## 0. Cible d'architecture

```
rag_builder/
├── core/              # cœur réutilisable (importable par API, worker, CLI, MCP)
│   ├── converters/    # ex-rag/converters.py, découpé + interface VisionDescriber/VideoTranscriber
│   ├── chunker.py     # ex-rag/chunker.py + tests + (code fences)
│   ├── images.py      # ImageStore multi-collections
│   ├── embeddings/    # interface Embedder : local_bge_m3 (défaut), gemini, mistral
│   ├── rerank.py      # cross-encoder local bge-reranker-v2-m3
│   ├── llm/           # interface LLMProvider (streaming) : mistral, anthropic, gemini, ollama
│   ├── store.py       # QdrantStore hybride (dense+sparse, delete-by-filter)
│   ├── retrieval.py   # HybridRetriever (Qdrant Query API) + rrf_fuse (inter-queries)
│   ├── rag_service.py # orchestration query (embed → search → rerank → generate)
│   └── config.py      # settings (pydantic-settings) + résolution par collection
├── db/                # SQLModel : users, collections, documents, jobs, feedback, settings
├── worker/            # boucle de traitement des jobs (ingestion asynchrone + progression)
├── api/               # FastAPI : REST + SSE + auth cookie + service statique front
├── mcp/               # serveur MCP (rag_query(collection, question)) — client de core
├── web/               # React + Vite + TS + Tailwind (4 pages), buildé statique
└── __main__.py        # CLI : ingest | ask | delete --collection X

storage/               # app.db (SQLite WAL), qdrant/ (mode local), collections/<c>/images/
eval/                  # golden.jsonl + run_eval.py
docker/                # Dockerfile(s), docker-compose.yml, Caddyfile, backup.sh
```

Le POC racine (`rag/`, `ingest.py`, `ask.py`, `mcp_server.py`) **reste fonctionnel
jusqu'à la fin de la Phase 1**, puis sera retiré (ou gardé en `legacy/`) une fois la
parité atteinte — **[VALIDER]** au moment venu.

### Décisions Phase 0 actées (voir `ETAT_DES_LIEUX.md` §9)
- **Nouveau dépôt propre** `github.com/Titouanos/RAG-DATA.git` (sans l'historique pollué du
  POC). Tâche 1.1 : init git propre + remote + commit initial ; `.webui_secret_key`
  gitignoré. ⚠️ Rappel utilisateur : **faire tourner** la clé `.webui_secret_key`.
- **Office legacy via LibreOffice headless** (converter `soffice`, pas de COM Windows).
- **Vidéo & YouTube retirés de la v1** → converters non portés (v2). Pas de
  `youtube-transcript-api`, pas de File API Gemini pour l'ingestion.
- **Génération testée via Gemini** en Phase 2 (clé existante) ; **Mistral = défaut cible**,
  câblé/basculé dès réception de la clé. Embeddings 100 % locaux (bge-m3) dès la Phase 1.

---

## Phase 1 — Cœur multi-collections (`rag_builder/core`)

**Objectif** : Qdrant hybride, embeddings bge-m3 locaux, suppression par `doc_id`,
réutilisation converters/chunker, CLI de validation.

### 1.1 Fondations projet
- `pyproject.toml` (build, deps, `ruff` + `pytest` config, `python-requires >=3.11`),
  lock des versions. `.env.example` étendu (QDRANT_MODE, provider LLM, clés).
- Corriger `.gitignore` (permettre `docs/*.md`, ignorer `*.log`, `.webui_secret_key`).
- `CLAUDE.md` racine (conventions, commandes run/test/lint, archi, décisions actées).
- Squelette `rag_builder/` + `__init__` typés + logging structuré centralisé.

### 1.2 Extraction chunker + images + converters
- Copier `chunker.py` dans `core/`, **ajouter tests unitaires** (hiérarchie titres,
  overlap, réparation URLs, protection `rag-image://`, chunks courts). *(R6 : détection
  code fences/tableaux = amélioration notée, parité POC d'abord.)*
- `ImageStore` multi-collections (chemin `collections/<c>/images/…`, validation `doc_id`).
- Découper `converters.py` par converter ; interface `VisionDescriber(bytes,mime)->str`
  injectée ; factoriser le cache Vision (clé = `hash(bytes)+modèle+prompt_version`, R7).
- **Legacy Office → converter LibreOffice headless** (remplace le COM Windows). **Vidéo &
  YouTube non portés** (v2). `VideoTranscriber` hors périmètre v1.

### 1.3 Embeddings locaux bge-m3
- Interface `Embedder` : `embed_documents`, `embed_query`, `dense_dim`, `model_id`,
  et **sparse** (dict indices→poids) pour bge-m3. Implémentations : `local_bge_m3`
  (défaut, via `fastembed`), `gemini` (portage de l'existant), `mistral`.
- Pré-téléchargement des poids dans un cache local (aucun download HuggingFace au runtime,
  exigence déploiement).

### 1.4 QdrantStore hybride
- `QdrantStore` : une collection Qdrant par collection RAG ; named vectors dense (1024,
  cosine) + sparse ; payload `{doc_id, source_name, page_or_section, chunk_index, headers,
  collection_id}`. IDs de points = **UUIDv5 déterministe** de `f"{doc_id}#{chunk_index}"`.
- `upsert`, `delete_by_doc_id` (filtre payload), `search` via Query API (`prefetch`
  dense+sparse + fusion RRF native), `count`, `create_collection(embedding_model, dims)`.
- Sélection `QDRANT_MODE=local|server` (même API) via env.
- Le **modèle d'embedding est figé par collection** (stocké en métadonnée collection ;
  changer = réindexation complète).

### 1.5 Reranker local
- `bge-reranker-v2-m3` (cross-encoder), top 20 → top 5, activable par collection ;
  spec : skip si `candidats ≤ top_k`, fallback tri-RRF sur échec.
- **Mesure de latence** (embed query / search / rerank) loguée ; **[VALIDER]** : si
  rerank CPU > 800 ms sur ce matériel, je le désactive par défaut, **mesures à l'appui**.

### 1.6 Orchestration + CLI
- `rag_service` : `ingest_document(collection, source)` (convert → chunk → embed → upsert),
  `query(collection, question)` (embed → search hybride → rerank → contexte),
  `delete_document(collection, doc_id)`. Déduplication par `content_hash`.
- CLI `python -m rag_builder ingest|ask|delete --collection X` (sans API/DB : Qdrant +
  un registre minimal de collections suffisent pour valider le cœur).

### DoD Phase 1
- `pytest` : chunking + upsert/delete/search hybride sur **corpus fixture** (petit PDF +
  markdown de test embarqués).
- **Latence de retrieval mesurée et affichée** (embed/search/rerank).
- **La suppression d'un doc fait disparaître ses chunks des résultats** (test dense+sparse).
- `ruff` OK, commits conventionnels, compte-rendu.

### Points [VALIDER] en Phase 1
- Choix `fastembed` pour dense+sparse+rerank (fallback `FlagEmbedding` si le reranker CPU
  n'y est pas supporté proprement) — je remonterai si blocage.
- Seuil de désactivation auto du rerank (>800 ms CPU) — je te présenterai les mesures.

---

## Phase 2 — API + worker (FastAPI, SQLite, jobs, SSE, auth)

### 2.1 Base de données (SQLModel, SQLite WAL, `storage/app.db`)
- Tables : `users` (login, hash argon2/bcrypt, rôle admin/user), `collections` (nom,
  description, embedding_model figé, params LLM/top_k/rerank/prompt système),
  `documents` (nom, hash, taille, statut, nb_chunks, dates), `jobs` (type, statut,
  progression par étape, message d'erreur), `feedback` (👍/👎 lié user + requête),
  `settings`. Migrations (`alembic` ou création au boot) — **[VALIDER]**.

### 2.2 Worker d'ingestion asynchrone
- Process/thread lancé avec l'API ; consomme les `jobs` `pending`.
- Progression par étape exposée (`parsing 3/12 pages`, `embedding 240/800 chunks`).
- Au **redémarrage**, les jobs `running` repassent `pending`. Déduplication : hash déjà
  présent → job `skipped`. **PDF corrompu/protégé → job `failed` + message clair**, jamais
  de crash. Taille d'upload max configurable (défaut 100 Mo).
- **Détection pages PDF sans couche texte** → suggestion OCR (OCR lui-même en Phase 4 ou
  ici si peu coûteux — **[VALIDER]** le placement).

### 2.3 API REST + SSE
- CRUD `/collections` ; `/collections/{id}/documents` (upload multipart, liste, delete) ;
  `/jobs/{id}` ; `POST /collections/{id}/query` en **SSE** (event `sources`, puis `token`*,
  puis `done`) ; `/health` ; endpoint images par collection.
- **LLMProvider streaming** branché (Mistral défaut) ; citations `[n]` mappées aux chunks.
- Prompt de génération : **honnêteté** (répondre uniquement à partir des extraits, dire
  quand l'info manque) + **résistance injection** (documents = données, pas instructions).

### 2.4 Auth
- Comptes locaux, mots de passe hashés (argon2/bcrypt), **sessions cookie httpOnly**,
  rôles admin/user. L'admin crée les comptes ; création/suppression de collections
  réservée aux admins par défaut (configurable). Bootstrap du premier admin — **[VALIDER]**
  (variable d'env / commande CLI `create-admin`).

### DoD Phase 2
- **Scénario complet documenté en `curl`/httpie** : créer collection → upload PDF → suivre
  le job → question streamée avec sources → supprimer un doc → vérifier que ses chunks ont
  disparu. Tests API (`httpx`/`TestClient`). `ruff` OK, CR.

---

## Phase 3 — Frontend (React + Vite + TS + Tailwind)

- **Page Collections** : liste, création (nom, description, presets), suppression + confirm.
- **Page Détail collection** : tableau documents (statut, chunks, date), upload drag & drop
  multi-fichiers, **progression jobs temps réel**, suppression doc, bouton réindexer.
- **Page Chat** : streaming token par token, **panneau sources cliquable** (fichier, page,
  extrait, score), affichage des images `rag-image://`, feedback 👍/👎 en base.
- **Page Paramètres collection** : provider/modèle LLM, top_k, rerank on/off, prompt système
  ; l'UI **explique** que changer l'embedding = réindexation ; **suggère l'OCR** quand des
  pages sans texte sont détectées.
- Front **buildé et servi en statique par FastAPI** en prod.

### DoD Phase 3
- **Parcours complet à la souris sans toucher au terminal.** Lint front, CR.

---

## Phase 4 — Providers, éval, packaging, MCP

### 4.1 Providers LLM
- Adapters `mistral` (défaut `mistral-large-latest`), `anthropic` (Haiku 4.5 / Sonnet —
  **[VALIDER]** ID exact), `gemini` (existant), `ollama` (100 % local). Retries + backoff,
  erreurs typées. Config par collection surchargeant l'env.

### 4.2 Évaluation (`eval/`)
- **Format `golden.jsonl`** défini + **jeu d'exemple** fourni (tu fourniras les ~20
  questions réelles). Champs proposés : `{ "question", "expected_doc", "expected_keywords":
  [...], "collection" }`.
- `run_eval.py` : **recall@5** et **MRR** du retrieval + **taux de présence des mots-clés**
  dans la réponse générée ; sortie **tableau + JSON**. Sert de non-régression.

### 4.3 OCR (si non fait en 2.2)
- `ocrmypdf`/Tesseract `fra+eng`, **désactivé par défaut**, activable par collection.

### 4.4 Packaging & déploiement
- **`docker compose up`** : api+worker, qdrant (mode server), reverse proxy TLS (Caddy),
  front buildé. Modèles locaux **pré-embarqués** (image ou volume cache) → **un seul flux
  sortant au runtime** (API du provider LLM). Mode **dev sans Docker** (Qdrant embarqué).
- **Script de sauvegarde** (SQLite + snapshot Qdrant) prêt pour cron.
- **README complet** (les deux modes). **CLAUDE.md** à jour.

### 4.5 MCP
- Serveur MCP adapté : `rag_query(collection, question)`, client de `rag_builder/core` ;
  transport streamable-http natif ; conserver la docstring-prompt empirique.

### DoD Phase 4
- **`docker compose up` sur machine neuve → démo complète.** `run_eval.py` produit un
  rapport. Tests verts, `ruff` OK, CR.

---

## Hors scope v1 (pistes v2 — à noter dans CLAUDE.md)

SSO Entra ID (OIDC) · permissions fines par collection/utilisateur · connecteurs
SharePoint/Drive automatiques · fine-tuning · agents multi-étapes · cache sémantique des
réponses · montée en charge > ~50 utilisateurs simultanés.

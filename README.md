# RAG Builder

Plateforme de **RAG multi-collections** : on crée une collection, on y dépose des documents
(PDF en priorité, Office, HTML, MindManager), le système construit un index hybride, on
interroge la collection en chat avec réponse streamée et **sources citées**, et on peut
ajouter/retirer des documents **sans tout réindexer**.

- **Stockage** : Qdrant hybride (dense + sparse dans le même point, suppression atomique par
  document).
- **Embeddings** : `BAAI/bge-m3` en local (multilingue FR/EN, 1024 dims, dense + sparse),
  100 % CPU, aucun appel externe pour l'indexation.
- **Génération** : interface multi-provider streamée (Gemini en Phase 1 ; Mistral par défaut
  cible, Anthropic/Ollama en option).

> État : **Phase 2** — cœur multi-collections + **API FastAPI** (REST + SSE) + **worker**
> d'ingestion asynchrone + **SQLite** + **auth** comptes locaux. Le frontend et le packaging
> Docker arrivent aux phases suivantes — voir [`docs/PLAN.md`](docs/PLAN.md).

## Prérequis

- Python **3.11+**
- ~5 Go d'espace disque pour le cache des modèles locaux (bge-m3 + reranker)
- Facultatif : `soffice` (LibreOffice) dans le PATH pour convertir les Office legacy
  (`.doc/.xls/.ppt`)

## Installation (mode dev, sans Docker — Qdrant embarqué)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[gemini,dev]"

# Pré-télécharger les modèles locaux une fois (ensuite tout tourne hors-ligne) :
export HF_HOME="$PWD/storage/models_cache"
python -c "from huggingface_hub import snapshot_download as d; \
  d('BAAI/bge-m3', ignore_patterns=['*.onnx','onnx/*']); \
  d('BAAI/bge-reranker-v2-m3', ignore_patterns=['*.onnx','onnx/*'])"

cp .env.example .env    # renseigne GEMINI_API_KEY si tu veux la génération
```

## Utilisation (CLI Phase 1)

```bash
# 1. Créer une collection (le modèle d'embedding est figé à la création)
python -m rag_builder create-collection ma_doc --description "Doc interne"

# 2. Ingérer des documents (chemins explicites, ou tout le dossier data/)
python -m rag_builder ingest --collection ma_doc chemin/vers/fichier.pdf
python -m rag_builder ingest --collection ma_doc          # tout data/

# 3. Interroger (retrieval + génération streamée ; -v montre les extraits et la latence)
python -m rag_builder ask --collection ma_doc "Comment faire X ?"
python -m rag_builder ask --collection ma_doc "..." --no-generate -v   # retrieval seul

# 4. Retirer un document (ses chunks disparaissent immédiatement des résultats)
python -m rag_builder delete --collection ma_doc --source fichier.pdf

# État / liste
python -m rag_builder stats --collection ma_doc
python -m rag_builder list-collections
```

## API web (Phase 2)

```bash
pip install -e ".[gemini,api,dev]"
python -m rag_builder create-user admin --admin      # 1er compte + crée storage/app.db
python -m rag_builder serve --host 127.0.0.1 --port 8000
```

REST + SSE, auth par cookie de session (rôles admin/user), worker d'ingestion asynchrone
avec progression, requête streamée (`sources` → `token*` → `done`). Parcours complet en
`curl` : [`docs/API_SCENARIO.md`](docs/API_SCENARIO.md).

## Développement

```bash
ruff check rag_builder tests          # lint (zéro erreur attendu)
pytest tests/ -m "not slow"           # tests rapides
pytest tests/                         # + intégration bge-m3 (charge les modèles)
```

Conventions, architecture détaillée et décisions : [`CLAUDE.md`](CLAUDE.md).

## Formats supportés (v1)

PDF (PyMuPDF), Office moderne `.docx/.pptx/.xlsx` et HTML/txt/md/csv (markitdown), Office
legacy `.doc/.xls/.ppt` (LibreOffice headless), MindManager `.mmap`. Vidéo/YouTube et OCR
sont hors périmètre v1 (cf. `docs/PLAN.md`).

## Déploiement

Le déploiement Docker complet (api + worker + Qdrant serveur + reverse proxy TLS + front
buildé, modèles pré-embarqués, script de sauvegarde) est livré en **Phase 4**. En mode dev,
Qdrant tourne en embarqué (aucun service à lancer).

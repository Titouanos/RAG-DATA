# État des lieux — POC `rag_poc` → cible RAG Builder

> Phase 0 de la mission. Ce document décrit l'architecture actuelle du POC, statue
> module par module sur ce qui est **réutilisé tel quel / adapté / remplacé** vers
> `rag_builder/`, et liste les **risques** identifiés (avec `fichier:ligne`).
> Il a été établi par lecture intégrale du code (racine + `rag/`, ~6 200 lignes) et
> par trois vérifications adversariales (désynchronisation BM25, audit des dépendances,
> hygiène/sécurité).

Date : 2026-07-07 · Commit analysé : `3273f63` · Python cible : 3.11+

---

## 1. Vue d'ensemble

Le POC est un RAG **mono-collection** local, entièrement construit sur **Gemini**
(embeddings + génération + vision + reranking + transcription vidéo) et **ChromaDB**
embarqué, piloté par CLI et exposé à OpenWebUI via un serveur **MCP**.

```
Entrées (data/)                Cœur (rag/)                        Sorties
──────────────       ────────────────────────────────    ─────────────────────
PDF, Office,   ─┐    converters.py  →  chunker.py         ingest.py  (CLI ingest)
.mmap, vidéos,  ├──► (markdown+images)  (markdown-aware)  ask.py     (CLI Q&A)
YouTube, HTML  ─┘         │                   │           mcp_server.py (MCP+HTTP
                          ▼                   ▼                        images)
                    images.py            embeddings.py (Gemini)
                    (ImageStore,             │
                     rag-image://)           ▼
                                        store.py (ChromaDB) + retrieval.py
                                        (BM25 picklé + RRF) ← rate_limit.py
                                                 │
                                            llm.py (génération + rerank +
                                            query expansion, tout Gemini)
                          orchestration : pipeline.py
```

**Chiffres** : `converters.py` 2448 l., `llm.py` 619 l., `pipeline.py` 532 l.,
`rate_limit.py` 357 l., `retrieval.py` 322 l., `chunker.py` 310 l., `store.py` 226 l.
Aucun test. Aucun `pyproject.toml`/lock. Aucun Dockerfile.

**Bonne nouvelle pour la migration** : les points d'extension sont déjà propres.
Le store vectoriel (`ChromaStore`), l'embedder (`GeminiEmbedder`), le BM25 (`BM25Store`)
et les capacités LLM (`GeminiClient`, vision/vidéo/rerank/expansion) sont **isolés**
et **injectés** par le pipeline. Les converters reçoivent des *callables* opaques
(`vision_describer(bytes, mime)->str`, `transcriber(path)->str`) : ils n'importent
jamais `google-genai`. Les CLI sont provider-agnostiques à ~95 %.

---

## 2. Inventaire module par module

| Module | Rôle | Verdict migration |
|---|---|---|
| `rag/converters.py` | Sources → markdown unifié + images | **Adapter** (COM Office & vidéo/YT à trancher) |
| `rag/chunker.py` | Chunking markdown-aware | **Conserver** (+ tests, + code fences) |
| `rag/images.py` | ImageStore content-addressed + `rag-image://` | **Conserver / adapter** (dimension collection) |
| `rag/store.py` | ChromaDB mono-collection + DocState JSON | **Remplacer** (Qdrant + SQLite) |
| `rag/retrieval.py` | BM25 picklé + fusion RRF + HybridRetriever | **Remplacer** le stockage, **conserver** RRF |
| `rag/embeddings.py` | Embeddings Gemini (matriochka 1536) | **Adapter** → impl. `gemini` de l'interface `Embedder` |
| `rag/llm.py` | Génération + rerank + expansion + vision + vidéo (Gemini) | **Remplacer** (LLMProvider streaming) / **adapter** (prompts, vision, vidéo) |
| `rag/rate_limit.py` | Rate limiter RPM/RPD Gemini | **Adapter** (par provider, multi-process) |
| `rag/pipeline.py` | Orchestration ingest + query | **Remplacer** (worker + core services) |
| `rag/config.py` | Chargement `config.yaml` + `.env` | **Adapter** (config par collection en SQLite) |
| `ingest.py` / `ask.py` | CLI | **Adapter** → `python -m rag_builder …` |
| `mcp_server.py` | Serveur MCP stdio + HTTP images | **Adapter** (cible une collection, intégré à l'API) |

---

## 3. À CONSERVER tel quel (ou quasi)

Ces briques sont provider-agnostiques et portables Linux — on les copie/adapte dans
`rag_builder/core` en ajoutant des tests (aucun n'existe aujourd'hui).

- **`MarkdownChunker`** (`rag/chunker.py`) — 100 % stdlib. Découpe par hiérarchie de
  titres ATX, re-split par taille avec overlap, **injection du chemin parent**
  `[Doc > H1 > H2]` en préfixe (`chunker.py:73-77`), fusion des chunks courts,
  réparation des URLs coupées (`_repair_split_urls`, `chunker.py:180-223`) et
  protection des réfs `rag-image://` dans l'overlap (`chunker.py:270-276`). Format de
  préfixe **à préserver à l'identique** (visible par les embeddings et le LLM).
- **`rrf_fuse`** (`retrieval.py:139-149`) — Reciprocal Rank Fusion pure, formule
  standard `Σ 1/(k + rang)`, poids égaux. Reste utile pour la **fusion inter-queries**
  du multi-query (la fusion dense+sparse intra-query devient native Qdrant).
- **`ImageStore` + `StoredImage` + `INTERNAL_SCHEME`** (`rag/images.py`) — stockage
  content-addressed idempotent (`sha256(bytes)[:20]`), garde anti path-traversal dans
  `resolve()` (`images.py:92-103`). À adapter uniquement pour la **dimension collection**
  dans le chemin (voir §4).
- **Converters PDF / Office OOXML / MindManager / YouTube / Markitdown** — la logique
  métier (PyMuPDF, parsing `.mmap` multi-versions, `AlternateImages` `mmarch://`,
  détection par magic bytes `converters.py:1828-1852`, extraction ZIP `word/media/`…)
  est **portable Linux** et rare/précieuse. Le modèle `ConvertedDoc` + `Protocol Converter`
  (`converters.py:31-47`) est une abstraction propre à transposer (en ajoutant `collection_id`).
- **Caches par hash** — Vision (clé = `sha256(bytes)`, JSON), vidéo (empreinte
  nom+taille+mtime), Office (conversion legacy). Pattern sain, à factoriser mais garder.
  ⚠️ *Réserve* : la clé de cache Vision n'inclut ni le modèle ni le prompt → un
  changement de modèle vision ne rafraîchit jamais les descriptions (voir §5).
- **Conventions produit à préserver bout en bout** :
  - Citations `[Source N]` mappées à `sources = [{num, title, source, type, chunk_id}]`
    (`llm.py:602-608`) — consommées par le MCP et l'UI.
  - Bloc image `*description riche indexée*\n\n![alt ≤80c](rag-image://<doc_id>/<hash>.<ext>)`
    (`converters.py:115-152`) + règle « recopier exactement, ne jamais inventer »
    (`llm.py:542-546`).
  - Durcissement de l'alt-text (`_sanitize_alt_text`, `converters.py:69-112`).
  - La **docstring-prompt** de l'outil MCP `rag_query` (`mcp_server.py:152-186`) : savoir
    empirique sur les questions énumératives, la fidélité des URLs, la section
    « Ressources complémentaires ».
- **Squelettes CLI** (argparse, gestion d'erreur config, codes de sortie 1/2) et
  l'endpoint FastAPI `/images/{path}` + `/health` + mapping extension→MIME
  (`mcp_server.py:94-129`) — fusionnera naturellement dans l'API.

---

## 4. À ADAPTER

- **`ConvertedDoc` / `doc_id`** — aujourd'hui `doc_id = file_<sha256(chemin_absolu)[:16]>`
  (`converters.py:55-57`), plus `mmap_<h>`, `video_<h>`, `yt_<id>`. **Problème** :
  dépend du **chemin absolu** → renommer/déplacer un fichier ou changer de machine
  invalide toute la déduplication et laisse des orphelins. **Cible** : `doc_id` géré
  par SQLite (UUID ou hash du contenu), `collection_id` explicite, `content_hash`
  conservé pour l'incrémental. Le contrat « `doc_id` stable + `content_hash` pour
  détecter les modifs » reste bon.
- **`GeminiEmbedder`** (`rag/embeddings.py`) → implémentation **`gemini`** de l'interface
  `Embedder`. Conserver le batching, l'asymétrie `embed_documents`/`embed_query`
  (task_types `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY`, propres à Gemini), la matriochka
  1536 + normalisation L2 (`_l2_normalize`, `embeddings.py:116-122`). La **valeur par
  défaut** devient `local_bge_m3` (dense 1024 + sparse). ⚠️ Corriger l'incohérence de
  troncature : commentaire « 8K chars » mais `MAX_CHARS_PER_CHUNK = 30_000`
  (`embeddings.py:66-69`), troncature silencieuse. Ajouter la vérif
  `len(embeddings) == len(batch)` (aujourd'hui absente, `embeddings.py:103-107`).
- **`HybridRetriever`** (`retrieval.py:162-322`) — garder la structure (`search`/`multi_search`,
  `RetrievedChunk`, injection embedder, fusion **inter-queries** à deux niveaux) mais
  **déléguer** le couple vecteur+lexical+RRF intra-query à la **Query API Qdrant**
  (`prefetch` dense+sparse, fusion RRF native). Supprimer l'accès à l'attribut privé
  `self.vector._collection` (`retrieval.py:244,296`, violation d'encapsulation) : les
  points Qdrant portent leur payload. Normaliser la gestion d'erreur `embed_query`
  (avalée en multi `retrieval.py:210-215`, propagée en single `retrieval.py:276`).
- **`rate_limit.py`** — le squelette (sliding-window RPM + compteur journalier + `stats()`)
  est réutilisable, mais : (1) le reset RPD « minuit Pacific Time » (`rate_limit.py:269-281`)
  et la table `DEFAULT_FREE_TIER_LIMITS` (`rate_limit.py:38-45`) sont **Gemini-only** →
  configurables par provider ; (2) `is_quota_error`/`extract_retry_delay`
  (`rate_limit.py:289-315`) reposent sur du **substring matching** sur `str(exc)` (fragile)
  → remplacer par les exceptions typées des SDK Mistral/Anthropic ; (3) la persistance
  JSON réécrite à chaque `acquire` **sans lock ni atomicité** (`rate_limit.py:238-257`) est
  **incompatible** avec l'architecture API + worker → SQLite. Pour les embeddings/rerank
  **locaux**, le rate limiter disparaît.
- **`ImageStore`** — introduire la **dimension collection** dans le chemin
  (`storage/collections/<coll>/images/<doc_id>/…`) et donc dans la réf `rag-image://`.
  ⚠️ **À décider avant migration** : la référence est gravée dans les chunks indexés
  dans Qdrant, on ne peut plus la changer après coup sans réindexer. Valider `doc_id`
  en entrée de `save()` (aujourd'hui aucune garde, un `../` échapperait à la racine,
  `images.py:60-61`).
- **Réécriture `rag-image://` → HTTP** — aujourd'hui dans `mcp_server.py:65-88` avec une
  base URL globale mutable forcée à `localhost` (bug en déploiement distant,
  `mcp_server.py:56-61`). À déplacer dans l'**API FastAPI** (endpoint par collection),
  URL de base configurable.
- **Prompts LLM** (`llm.py`) : `ANSWER_SYSTEM`/`ANSWER_PROMPT` (`llm.py:525-620`) sont bons
  (citations, honnêteté, règles images/URLs) mais (1) doivent passer en **streaming**
  (aucun streaming nulle part dans le POC, voir §5), (2) la 1ʳᵉ ligne (« assistant …
  DevOps/PLM/infra », `llm.py:525`) doit devenir **paramétrable par collection**.
  `QUERY_EXPANSION_PROMPT` (`llm.py:172-275`) embarque en dur le **vocabulaire Groupe
  Atlantic** (codes SAP Z002/Z007, Teamcenter, Creo…) + exemples few-shot →
  **template paramétrable par collection**, désactivé par défaut.
- **Vision / vidéo** : garder les prompts (`VISION_PROMPT`, `VIDEO_PROMPT`, français,
  anti-hallucination) mais brancher sur une interface `VisionDescriber` /
  `VideoTranscriber` multi-provider. Le filtre `_DECORATIVE_PATTERNS`
  (`converters.py:166-227`, ~60 phrases françaises calibrées sur Gemini) est **fragile** →
  à recalibrer ou remplacer par un champ JSON `is_decorative` demandé au modèle.
- **CLI / MCP** : `python -m rag_builder ingest|ask|delete --collection X` ;
  `rag_query(collection, question)` ; passer du transport stdio+`mcpo` au
  **streamable-http** natif de FastMCP maintenant qu'une API existe (supprime le hack
  thread daemon + `sleep(0.5)`).

---

## 5. À REMPLACER

- **`ChromaStore`** (`store.py:98-210`) → **`QdrantStore`**. La collection est codée en
  dur (`COLLECTION_NAME = "rag_chunks"`, `store.py:105`) — incompatible multi-collections.
  Cible : une collection Qdrant par collection RAG, **named vectors** dense+sparse,
  **suppression par filtre `doc_id`** (au lieu de tracker manuellement des listes de
  `chunk_ids`), payload natif (au lieu de metadata aplaties en primitives).
  ⚠️ Qdrant n'accepte que des **IDs `uint64` ou UUID** : convertir
  `f"{doc_id}_c{order:04d}"` (`pipeline.py:300`) en **UUIDv5 déterministe**, garder
  `doc_id`/`chunk_index` dans le payload.
- **`BM25Store` + pickle** (`retrieval.py:29-102`) → **vecteurs sparse natifs Qdrant**.
  Élimine : le `pickle.load` (**exécution de code arbitraire** si le fichier est altéré,
  `retrieval.py:71-75`), la **duplication de tout le corpus texte** dans le pickle, le
  **rebuild global O(corpus)** à chaque ingestion (`pipeline.py:346-349`, charge tout en
  RAM via `get_all`), et **structurellement** le risque de désynchronisation
  lexical/vectoriel (voir encadré ci-dessous).
- **`DocState` JSON** (`store.py:33-82`) → tables **SQLite/SQLModel** (`documents`, `jobs`…).
  Le JSON réécrit en entier à chaque `save`, sans verrou ni transaction, est inadapté à
  API + worker concurrents. `chunk_ids` devient superflu (suppression par filtre Qdrant) ;
  garder `content_hash` pour l'incrémental.
- **`GeminiReranker` + `RERANK_PROMPT`** (`llm.py:410-517`) → **cross-encoder local
  `bge-reranker-v2-m3`**. Le reranker LLM est lent (1 appel/requête), coûte du quota,
  dépend d'un parsing JSON fragile, et sa pondération `0.8·LLM + 0.2·RRF`
  (`llm.py:477-481`) est ad hoc. Conserver seulement comme **spécification du nouveau
  reranker** : skip si `candidats ≤ top_k` (`llm.py:447-449`) et fallback tri-RRF sur échec.
- **`GeminiClient` bloquant** (`llm.py:29-98`) → interface **`LLMProvider` streaming**
  (`mistral` défaut, `anthropic`, `gemini`, `ollama`). `generate_content` +
  `response.text` devient un **générateur de tokens** (SSE).
- **Query expansion sur le chemin critique** — désactivée par défaut (décision actée) :
  2 appels LLM avant génération = latence + quota. Code conservé en option par collection.

> ### ⚠️ Vérification adversariale — « le BM25 picklé se désynchronise à la suppression »
> **Nuance importante.** En fonctionnement **nominal**, il n'y a **pas** de désync : toute
> suppression/réingestion passe par `ingest()` qui modifie Chroma puis **reconstruit
> intégralement** le BM25 depuis Chroma en fin de run (`pipeline.py:346-349`). L'argument
> « impossible proprement avec le pickle » vaut surtout pour une **suppression unitaire**
> (qui **n'existe pas** dans le POC — la suppression est implicite : fichier disparu de
> `data/`, `pipeline.py:331-340`) et pour la **robustesse** :
> - **Crash entre le `delete` Chroma et le rebuild BM25** → `bm25.pkl` garde des chunks
>   supprimés, rechargé tel quel au démarrage **sans contrôle de cohérence** ; le fallback
>   `vec_map.get(cid) or bm25_map.get(cid)` (`retrieval.py:311`) fait alors **remonter un
>   chunk fantôme** avec texte périmé et `metadata={}` (donc `doc_id=''`, ce qui casse la
>   diversification `pipeline.py:411`).
> - **`ingest --reset` interrompu** : Chroma vidé mais `bm25.pkl` intact → résultats
>   majoritairement fantômes.
> - **Bug data-loss réel** : sur `QuotaExhaustedError` la boucle `break`
>   (`pipeline.py:236,279`) mais l'étape 3 s'exécute quand même et considère tous les docs
>   **non encore parcourus** comme « absents » → `delete` + `remove` du `doc_state`
>   (`pipeline.py:331-340`), **en contradiction** avec le message « les docs déjà indexés
>   sont préservés ». Idem sur échec transitoire de conversion.
>
> **Conclusion** : Qdrant hybride natif (dense+sparse dans le même point, delete-by-filter)
> **élimine structurellement** toute cette classe de problèmes. Décision confirmée — mais
> à présenter comme un gain de **robustesse/atomicité + suppression unitaire**, pas comme
> la correction d'une désync systématique en régime nominal.

---

## 6. Dépendances (audit)

- **Aucune dépendance fantôme** (tous les imports mappent à `requirements.txt`) et
  **aucune déclarée non utilisée**. **Aucune version pinée** : 14/14 en `>=`, dont
  `markitdown[all]>=0.0.1a3` (**pré-release alpha**) → build non reproductible.
- **Portabilité Linux** :
  - `pywin32` + COM Word/Excel/PowerPoint (`converters.py:388,419-420,1047-1048`) est
    **Windows-only** → sous Linux les `.doc/.xls/.ppt` **legacy** et l'enrichissement
    **screenshots PPTX** sont silencieusement perdus. Décision requise (voir §9-Q2).
  - La **transcription vidéo ne nécessite pas ffmpeg** : le fichier est uploadé brut au
    File API Gemini (`llm.py:307-345`). Aucun binaire système requis pour ce chemin.
  - `lxml`, `pymupdf`, `chromadb`, `markitdown` (via `magika`/`onnxruntime`) : wheels
    manylinux, aucun paquet système requis.
- **À AJOUTER pour la cible** (sans versions ici) : `qdrant-client`, **`fastembed`**
  (recommandé vs `sentence-transformers`/`FlagEmbedding` qui tirent `torch` ≈ 2 Go —
  pénalisant en CPU-only ; à valider : support cross-encoder pour le reranker),
  `sqlmodel` (→ `sqlalchemy`+`pydantic`), `python-multipart` (upload), `httpx`
  (client + `TestClient`), `mistralai`, `anthropic` ; **optionnels** `alembic`
  (migrations), `pydantic-settings`, `pytest`, `ruff`. **Hors pip** : LibreOffice
  headless (paquet système Docker) si les formats legacy doivent rester.
- **Redondance à surveiller** : `chromadb` et `markitdown` tirent déjà `onnxruntime`,
  comme `fastembed` → surveiller les conflits de versions au moment du lock. `chromadb`
  et `rank-bm25` deviennent obsolètes après migration Qdrant ; `mcp` reste.

---

## 7. Sécurité & hygiène (à traiter avant/pendant la migration)

- ⚠️ **`.webui_secret_key` versionné** (`git ls-files`, ajouté au commit initial
  `7f25f7b`). C'est la **clé de signature des JWT de session Open WebUI** (chaîne ASCII
  16 c.), écrite par une instance Open WebUI lancée depuis ce dossier — **committée par
  erreur**, aucun code du projet ne la lit. **Untrack + `.gitignore` ne suffit pas** :
  la valeur reste dans l'historique. → **à faire tourner** côté Open WebUI, et décider si
  on **réécrit l'historique** (voir §9-Q1).
- ⚠️ **Historique pollué** : le commit `7f25f7b` contient `ingest_full.log` (supprimé au
  commit suivant mais lisible via `git show`) qui **divulgue** ~68 noms de documents
  internes Teamcenter/PLM, des **chemins Windows nominatifs** (`C:\Users\ndeslauriers\…`)
  et l'email d'auteur `ndeslauriers@groupe-atlantic.com`. Pas de credential, mais fuite
  d'info interne si le repo est partagé.
- **Aucun autre secret en dur** : la clé Gemini vient exclusivement de `.env`
  (`config.py:97-104`, garde-fou anti-placeholder), `.env` est gitignoré. `git log -p`
  sur les 2 commits : aucun credential.
- **`.gitignore` bloque `docs/`** (`.gitignore:33`) → **ce fichier et `PLAN.md` ne seront
  pas versionnables** sans corriger le `.gitignore` (remplacer `docs/` par `docs/*` +
  `!docs/*.md`, ou restreindre les règles `*.pdf`/Office à `data/`). Les règles `*.pdf`,
  `*.docx`… (`.gitignore:34-40`) s'appliquent **partout**, pas seulement à `data/`.
  Ajouter aussi `*.log` et `.webui_secret_key`.
- **Serveur d'images** : bind `127.0.0.1:8765` par défaut mais **CORS `*`** et **aucune
  auth** (`mcp_server.py:103-127`) ; la chaîne `mcpo` recommandée (`README.md:208-215`)
  expose le RAG **sans auth** sur `0.0.0.0`. → restreindre CORS, auth via reverse proxy
  dans la cible.
- **Pas d'injection shell** (aucun `subprocess`/`os.system`/`eval`), path-traversal des
  images correctement géré (`images.py:92-103`). Risque résiduel **inhérent au RAG** :
  le contenu des documents part dans les prompts → **injection via documents**, à
  contrer par le prompt système (exigence actée).
- **`config.yaml` calibré Tier 1** (billing Gemini activé, `config.yaml:123-135`) → tracer
  la propriété/coût de la clé. Avec embeddings **locaux**, Gemini n'est plus requis pour
  l'indexation.

---

## 8. Risques transverses (synthèse priorisée)

| # | Risque | Où | Traitement cible |
|---|---|---|---|
| R1 | Data-loss sur quota/échec pendant l'ingest (suppression des docs non parcourus) | `pipeline.py:236,279,331-340` | Worker par job atomique ; jamais de suppression implicite globale |
| R2 | Chunks fantômes BM25 après crash (metadata vide, `doc_id=''`) | `retrieval.py:311` | Qdrant delete-by-filter (dense+sparse synchrones) |
| R3 | `pickle.load` du BM25 = exécution de code arbitraire | `retrieval.py:71-75` | Supprimé (sparse natif Qdrant) |
| R4 | `doc_id` = hash du **chemin absolu** → orphelins au déplacement | `converters.py:55-57` | ID géré en SQLite (UUID/hash contenu) |
| R5 | Formats Office legacy + screenshots PPTX **perdus sous Linux** | `converters.py:320-328,1037-1109` | LibreOffice headless ou abandon (Q2) |
| R6 | Chunker casse les **code fences** et **tableaux** (titres `#` dans un bloc, split sur lignes vides) | `chunker.py:21,142` | Détection de fences + tableaux (parité POC d'abord) |
| R7 | Cache Vision non invalidé au changement de modèle/prompt (clé = bytes seuls) | `converters.py:713,1141,1344,1863` | Clé = `hash(bytes)+modèle+version_prompt` |
| R8 | Aucun **streaming** dans tout le POC (réécriture de l'interface génération) | `llm.py:57-67,613-619` | `LLMProvider` async + SSE |
| R9 | Rate limiter non multi-process (JSON non atomique, itération hors lock) | `rate_limit.py:124,238-257` | SQLite / suppression pour local |
| R10 | Prompts (génération + expansion) **spécifiques Groupe Atlantic** | `llm.py:174-216,525` | Templates paramétrables par collection |
| R11 | `_DECORATIVE_PATTERNS` calibrés sur le style Gemini FR | `converters.py:166-227` | Champ `is_decorative` demandé au modèle |
| R12 | Alignement placeholders/images OOXML fragile (même image 2×, headers) | `converters.py:1268-1305` | Robustifier l'injection ordonnée |
| R13 | YouTube sans cache + titre bidon `YouTube — <id>` + bans IP datacenter | `converters.py:2128-2141` | Cache par `video_id`, oEmbed pour le titre (si conservé) |
| R14 | Boot du serveur charge tout le pipeline ; avec modèles locaux → plusieurs dizaines de s | `mcp_server.py:281` | Lazy-load / warm-up au démarrage du worker |
| R15 | Migration embeddings Gemini→bge-m3 = **réindexation totale** (dims + normalisation différentes) | `embeddings.py:198` | Repartir de zéro (aucun vecteur réutilisable) |

---

## 9. Décisions structurantes (Phase 0 — actées)

1. **Dépôt cible : nouveau repo propre** `https://github.com/Titouanos/RAG-DATA.git`.
   Résout d'emblée l'historique pollué (`.webui_secret_key`, `ingest_full.log`,
   chemins/email nominatifs, §7) : le nouveau dépôt part du working tree actuel **sans
   l'historique** de `wigowww/rag_poc`. `.webui_secret_key` sera gitignoré et non ajouté ;
   les logs internes ne sont pas dans le working tree. ⚠️ **Rappel** : la clé
   `.webui_secret_key` reste à **faire tourner côté Open WebUI** (elle a été exposée
   publiquement dans `wigowww/rag_poc`).
2. **Office legacy** : **LibreOffice headless** dans l'image Docker (`soffice`) pour
   convertir `.doc/.xls/.ppt` → OOXML et rendre les slides PPTX (remplace le COM Windows,
   R5). Le `LegacyOfficeConverter` COM est remplacé par un converter LibreOffice.
3. **Vidéo & YouTube : retirés de la v1** (piste v2). `VideoConverter` et `YouTubeConverter`
   ne sont **pas portés** → suppression du couplage File API Gemini pour l'ingestion et de
   la dépendance `youtube-transcript-api`. Focus documents (PDF prioritaire, Office, HTML,
   `.mmap`).
4. **Tests génération Phase 2** : **réutiliser la clé Gemini existante** comme provider LLM
   en attendant la clé Mistral. Mistral (`mistral-large-latest`) reste le **défaut cible** ;
   je le câble et bascule dès que la clé est fournie. *(Note : la génération peut utiliser
   Gemini via l'interface `LLMProvider` ; les embeddings restent 100 % locaux bge-m3.)*

Points **tranchés unilatéralement** (dans l'esprit du cahier des charges, pour info) :
`fastembed` plutôt que `sentence-transformers` (CPU léger) ; parité POC du chunker en
Phase 1 (améliorations code/tableaux en backlog) ; repartir d'un **corpus vide** (R15,
réindexation obligatoire) ; mapping Anthropic « Sonnet 4.6 » → ID réel (Sonnet 5 /
Opus 4.8) à confirmer au câblage provider (Phase 4).

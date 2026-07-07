"""Wrappers Gemini pour : génération, reranking, vision.

Tous les appels passent par google-genai (SDK unifié) et par le RateLimiter
pour respecter les quotas (notamment free tier).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List

from rag.rate_limit import (
    QuotaExhaustedError, RateLimiter, with_rate_limit_retry,
)
from rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client Gemini unique (réutilisé par tous les wrappers)
# ---------------------------------------------------------------------------


class GeminiClient:
    """Client Gemini partagé entre génération / rerank / vision.

    Tous les appels passent par un RateLimiter qui :
    - Attend proactivement avant chaque appel pour respecter les RPM.
    - Détecte les 429 et les retry en respectant le retryDelay renvoyé par Gemini.
    - Arrête net si la limite journalière (RPD) est atteinte.
    """

    def __init__(self, api_key: str, rate_limiter: RateLimiter | None = None):
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.rate_limiter = rate_limiter or RateLimiter()

    def generate_text(self, model: str, prompt: str,
                      temperature: float = 0.2,
                      max_tokens: int | None = None,
                      system: str | None = None) -> str:
        """Génère du texte simple depuis un prompt."""
        from google.genai import types

        config_kwargs = {"temperature": temperature}
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens
        if system:
            config_kwargs["system_instruction"] = system

        def _call():
            return self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )

        response = with_rate_limit_retry(
            _call, limiter=self.rate_limiter, model=model,
        )
        return response.text or ""

    def generate_vision(self, model: str, prompt: str,
                        image_bytes: bytes, mime_type: str,
                        temperature: float = 0.1,
                        max_output_tokens: int = 800) -> str:
        """Appel multimodal (image + prompt).

        max_output_tokens limite à ~600 mots la description. C'est une
        protection critique contre les boucles infinies de Gemini sur les
        images bruitées (arbres CAO avec 100+ features, listes de numéros,
        etc.) qui peuvent générer des dizaines de milliers de tokens.
        """
        from google.genai import types

        def _call():
            return self.client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )

        response = with_rate_limit_retry(
            _call, limiter=self.rate_limiter, model=model,
        )
        return response.text or ""


# ---------------------------------------------------------------------------
# Vision : décrire les screenshots de mindmaps
# ---------------------------------------------------------------------------


VISION_PROMPT = (
    "Décris cette capture d'écran de manière **factuelle et détaillée** pour qu'elle soit "
    "exploitable dans un système de recherche documentaire.\n\n"
    "Inclus :\n"
    "- Le logiciel ou l'interface visible\n"
    "- Les textes lisibles (menus, boutons, messages, valeurs saisies)\n"
    "- Les éléments cliqués, encadrés ou mis en évidence\n"
    "- Les messages d'erreur, codes, noms de champs\n"
    "- La nature de l'action illustrée (configuration, diagnostic, erreur...)\n\n"
    "**Cas particulier — grille pédagogique avec quelques cellules nommées** : "
    "si l'image contient une grille structurée de **moins de 25 cellules** où "
    "chaque cellule a un libellé court (titre + code), liste explicitement chaque "
    "cellule avec son libellé et son code. Exemple : pour une slide « TYPES "
    "ARTICLES » montrant 13 boîtes (Produits Finis Z001, Pièces Détachées Z003, "
    "etc.), liste les 13.\n\n"
    "**LIMITES STRICTES — toujours respecter** :\n"
    "- Maximum **300 mots** au total. Si le contenu dépasse, fais un résumé.\n"
    "- Ne jamais énumérer des séquences de nombres (1, 2, 3, 4...) sauf si "
    "  ce sont des codes/identifiants spécifiques visibles à l'écran.\n"
    "- Pour les arborescences ou listes très longues (>30 items), donne juste "
    "  la structure générale + 5-10 exemples représentatifs.\n"
    "- N'invente rien : si quelque chose est illisible, dis-le simplement."
)


def make_vision_describer(client: GeminiClient, model: str):
    """Retourne une fonction callable(image_bytes, mime) -> description.

    Implémente un retry exponentiel sur les erreurs 503 (UNAVAILABLE) qui
    surviennent quand Gemini est temporairement saturé. C'est important
    pour éviter de perdre des images à chaque ingestion : une 503 est
    généralement transitoire et passe au bout de quelques secondes.
    """
    import time

    def describe(image_bytes: bytes, mime: str) -> str:
        # Retry exponentiel : 0s, 5s, 15s, 30s = 4 tentatives max
        delays = [0, 5, 15, 30]
        last_exc = None
        for i, delay in enumerate(delays):
            if delay > 0:
                logger.info("Vision retry %d/%d après %ds...", i, len(delays) - 1, delay)
                time.sleep(delay)
            try:
                return client.generate_vision(model, VISION_PROMPT, image_bytes, mime).strip()
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                # Retry uniquement pour 503/UNAVAILABLE (saturation transitoire)
                # Pas pour 400 (bad request, ne sera jamais résolu)
                if "503" in exc_str or "UNAVAILABLE" in exc_str:
                    logger.warning("Vision saturé (503), retry...")
                    continue
                # Autre erreur : pas la peine de retry
                logger.warning("Vision a échoué : %s", exc)
                return ""
        # Tous les retries ont échoué
        logger.warning("Vision a échoué après %d tentatives : %s", len(delays), last_exc)
        return ""
    return describe


# ---------------------------------------------------------------------------
# Multi-query expansion
# ---------------------------------------------------------------------------

QUERY_EXPANSION_PROMPT = """Tu génères des sous-queries pour une recherche dans une base documentaire technique du Groupe Atlantic : Teamcenter (PLM Siemens), SAP, Creo (CAO PTC), infrastructure IT, méthodologies internes (PDM/PLM, gestion d'articles, nomenclatures, workflows).

Vocabulaire métier à connaître :
- Articles SAP types : Z002 (Produit Fini), Z007 (Matière première), HALB (Semi-fini), HAWA (Achat), HERS (Article Fabricant), VERP (Emballage), DSE (Sous-ensembles)
- Nomenclatures : XBoM, MBoM, CADBoM, EBoM
- Teamcenter : IPEM, Transfer Manager, ECN, Workflow, Article, Révision, Dataset, JT
- Creo : PRT, ASM, DRW, FDU (Fonction Définie Utilisateur)
- Synonymes : PLM ↔ Teamcenter ↔ TC, CAO ↔ Creo ↔ Pro/E

Question utilisateur : "{question}"

Génère 3 sous-queries qui DÉCOMPOSENT la question en angles ORTHOGONAUX (pas de simples reformulations !). L'objectif : remonter des chunks que la query originale raterait parce qu'ils utilisent un autre vocabulaire.

Stratégies à privilégier :
1. **Décomposition par catégories métier** : si la question est générale ("types d'articles SAP"), génère des queries pour chaque catégorie connue (Z002, Z007, HALB, etc.).
2. **Utilisation de synonymes/codes** : si la question utilise un terme générique, remplace-le par les codes/acronymes spécifiques du métier.
3. **Concepts connexes** : évite de répéter les mêmes mots-clés que la query originale.

Règles strictes :
- Ne JAMAIS reformuler simplement (ex: "types articles SAP" est interdit pour une question "types d'articles dans SAP")
- Chaque sous-query doit cibler un sous-thème DIFFÉRENT
- Format : 3 lignes, sans numérotation, sans tirets, sans préambule
- 3-7 mots par sous-query
- Reste en français

EXEMPLES :

Question : "c'est quoi les types d'articles dans SAP"
Réponse :
matière première Z007 sourcing
produit fini Z002 codification
semi-fini HALB fabrication

Question : "qu'est-ce qu'un PLM"
Réponse :
bénéfices PLM unicité traçabilité
gestion cycle vie produit Teamcenter
data vault coffre-fort produit

Question : "comment importer une pièce dans Teamcenter"
Réponse :
sauvegarde PRT DRW IPEM
intégration Creo Transfer Manager
migration manuelle données CAO

Maintenant pour la question utilisateur, génère 3 sous-queries DÉCOMPOSÉES (pas reformulées) :"""


def make_query_expander(client: GeminiClient, model: str):
    """Retourne une fonction expand_query(question) -> list[query].

    La première query retournée est toujours la question originale.
    Les suivantes sont des sous-queries générées par le LLM pour enrichir
    le retrieval.

    Implémente un retry sur les 503 (saturation Gemini transitoire) car
    la query expansion est critique pour la qualité du retrieval — on ne
    veut pas la perdre à chaque pic de charge.

    Si toutes les tentatives échouent, retourne juste [question] pour ne pas
    casser le pipeline (fallback graceful sur retrieval simple).
    """
    def expand(question: str) -> list[str]:
        question = (question or "").strip()
        if not question:
            return []
        prompt = QUERY_EXPANSION_PROMPT.format(question=question)

        # Retry 503 : 0s, 3s, 8s = 3 tentatives
        delays = [0, 3, 8]
        last_exc = None
        raw = None
        for i, delay in enumerate(delays):
            if delay > 0:
                logger.info("Query expansion retry %d/%d après %ds...",
                            i, len(delays) - 1, delay)
                time.sleep(delay)
            try:
                raw = client.generate_text(model, prompt, temperature=0.3, max_tokens=200)
                break
            except Exception as exc:
                last_exc = exc
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    continue
                # Autre erreur : abandon immédiat
                logger.warning("Query expansion a échoué : %s", exc)
                return [question]

        if raw is None:
            logger.warning("Query expansion a échoué après %d tentatives : %s",
                           len(delays), last_exc)
            return [question]

        # Parser les lignes
        sub_queries = []
        for line in raw.split("\n"):
            line = line.strip()
            line = re.sub(r"^[\d\.\-\*\)]+\s*", "", line)
            line = line.strip(" \"'")
            if line and len(line) >= 3:
                sub_queries.append(line)
        sub_queries = sub_queries[:3]
        return [question] + sub_queries
    return expand


# ---------------------------------------------------------------------------
# Vidéo : transcription + description visuelle via Gemini (natif, pas de ffmpeg)
# ---------------------------------------------------------------------------


VIDEO_PROMPT = """Tu vas analyser cette vidéo (probablement un tutoriel technique, une démo d'outil \
ou une explication métier) et produire un document markdown directement exploitable par un système \
de recherche documentaire (RAG).

**Format strict :**
1. Un titre H1 `# Titre décrivant le sujet de la vidéo`
2. Des sections H2 découpées par tranches d'environ 60 secondes, avec le timestamp en tête :
   `## [MM:SS] Titre court de la section`
3. Dans chaque section :
   - La **transcription** des paroles prononcées (en français si c'est la langue parlée).
   - Une **description factuelle** des éléments visuels pertinents : logiciel ou interface montrée, \
menus et boutons visibles, commandes tapées, valeurs saisies, messages d'erreur affichés, étapes \
réalisées à l'écran.

**Règles impératives :**
- Préserve **exactement** les commandes, chemins de fichiers, noms de variables, URLs, codes d'erreur \
et termes techniques tels qu'ils apparaissent ou sont prononcés.
- Si un logiciel précis est à l'écran, nomme-le.
- Si une portion est inaudible ou illisible, marque-la `[inaudible]` ou `[illisible]`.
- **N'invente rien**. Mieux vaut dire "je ne vois pas clairement" que combler un blanc.
- Ne fais aucun préambule ni commentaire méta. Commence directement par le titre H1."""


def make_video_transcriber(client: GeminiClient, model: str,
                            processing_timeout: int = 600,
                            poll_interval: int = 5):
    """Retourne une fonction callable(video_path) -> markdown.

    Utilise le File API de Gemini pour uploader la vidéo, attend qu'elle soit traitée,
    appelle le modèle en multimodal, puis supprime le fichier côté Gemini.

    Note : le SDK google-genai a un bug où les noms de fichiers contenant des
    caractères non-ASCII (accents, etc.) plantent à l'upload car ils sont passés
    dans un header HTTP. On contourne en copiant le fichier vers un nom temporaire
    ASCII-safe avant l'upload.
    """
    import shutil
    import tempfile
    import time
    import uuid
    from google.genai import types

    def _is_ascii_safe(name: str) -> bool:
        try:
            name.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    def transcribe(video_path) -> str:
        from pathlib import Path
        path = Path(video_path)

        # Bug google-genai : si le nom contient des accents/non-ASCII,
        # on copie d'abord vers un fichier temporaire avec un nom safe.
        upload_path = path
        temp_copy = None
        if not _is_ascii_safe(path.name):
            safe_name = f"video_{uuid.uuid4().hex[:12]}{path.suffix}"
            temp_copy = Path(tempfile.gettempdir()) / safe_name
            logger.debug("Nom de fichier non-ASCII (%s), copie vers %s pour upload",
                         path.name, temp_copy.name)
            shutil.copy2(path, temp_copy)
            upload_path = temp_copy

        # 1. Upload via File API (pas compté dans RPM/RPD du modèle)
        logger.info("Upload de %s vers Gemini Files API...", path.name)
        uploaded = None
        try:
            uploaded = client.client.files.upload(file=str(upload_path))
        finally:
            # On peut supprimer la copie temp dès l'upload terminé
            if temp_copy and temp_copy.exists():
                try:
                    temp_copy.unlink()
                except Exception:
                    pass

        try:
            # 2. Attendre que Gemini ait fini de traiter la vidéo
            start = time.time()
            while True:
                state = getattr(uploaded.state, "name", str(uploaded.state))
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise RuntimeError(
                        f"Gemini a échoué à traiter la vidéo {path.name}"
                    )
                if time.time() - start > processing_timeout:
                    raise TimeoutError(
                        f"Timeout ({processing_timeout}s) en attendant "
                        f"le traitement de {path.name}"
                    )
                logger.info("  vidéo en cours de traitement par Gemini...")
                time.sleep(poll_interval)
                uploaded = client.client.files.get(name=uploaded.name)

            # 3. Génération (rate-limited via with_rate_limit_retry)
            logger.info("Transcription en cours (%s)...", model)

            def _call():
                return client.client.models.generate_content(
                    model=model,
                    contents=[uploaded, VIDEO_PROMPT],
                    config=types.GenerateContentConfig(temperature=0.1),
                )

            response = with_rate_limit_retry(
                _call, limiter=client.rate_limiter, model=model,
            )
            return (response.text or "").strip()
        finally:
            # 4. Cleanup côté Gemini
            try:
                client.client.files.delete(name=uploaded.name)
            except Exception as exc:
                logger.debug("Cleanup du fichier Gemini a échoué : %s", exc)

    return transcribe


# ---------------------------------------------------------------------------
# Reranker : réordonne les candidats avec un LLM
# ---------------------------------------------------------------------------


RERANK_PROMPT = """Tu es un classificateur de pertinence pour un système de recherche documentaire.

Question de l'utilisateur :
<question>
{query}
</question>

Voici {n} passages candidats. Pour chacun, attribue un score de pertinence entre 0 et 10 :
- 10 = passage qui répond directement à la question
- 7-9 = passage très pertinent, contient des infos utiles
- 4-6 = passage partiellement lié
- 1-3 = passage vaguement lié
- 0 = passage hors-sujet

Réponds **uniquement** en JSON valide, un tableau d'objets {{\"id\": <int>, \"score\": <int>}}, sans texte autour.

Passages :
{passages}"""


class GeminiReranker:
    """Rerank batch des candidats via Gemini.

    On envoie les N candidats en une seule requête (économise les appels) et
    on demande un JSON avec les scores.
    """

    def __init__(self, client: GeminiClient, model: str,
                 max_chunk_chars: int = 1200):
        self.client = client
        self.model = model
        self.max_chunk_chars = max_chunk_chars

    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if len(candidates) <= top_k:
            # pas la peine de reranker, mais on trie quand même
            return sorted(candidates, key=lambda c: c.fused_score, reverse=True)[:top_k]

        # Tronquer les chunks pour économiser les tokens
        passages_txt = []
        for i, c in enumerate(candidates):
            snippet = c.text
            if len(snippet) > self.max_chunk_chars:
                snippet = snippet[:self.max_chunk_chars] + " [...]"
            passages_txt.append(f"<passage id=\"{i}\">\n{snippet}\n</passage>")
        passages_joined = "\n".join(passages_txt)

        prompt = RERANK_PROMPT.format(
            query=query, n=len(candidates), passages=passages_joined,
        )

        try:
            raw = self.client.generate_text(
                self.model, prompt, temperature=0.0, max_tokens=2000,
            )
        except Exception as exc:
            logger.warning("Reranker a échoué : %s — fallback sur scores RRF", exc)
            return sorted(candidates, key=lambda c: c.fused_score, reverse=True)[:top_k]

        scores = _parse_rerank_json(raw, len(candidates))

        # Combine les scores LLM avec le score RRF pour robustesse
        # (si le LLM met 0 partout, on garde l'ordre RRF)
        scored = []
        for i, c in enumerate(candidates):
            llm_score = scores.get(i, 0) / 10.0  # normalise 0-1
            # pondération : 80% LLM, 20% RRF
            combined = 0.8 * llm_score + 0.2 * min(c.fused_score * 10, 1.0)
            scored.append((combined, c))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [c for _, c in scored[:top_k]]


def _parse_rerank_json(text: str, n: int) -> dict[int, int]:
    """Parse la sortie JSON du reranker avec tolérance."""
    # Retirer les blocs ```json éventuels
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Chercher un tableau JSON dans le texte
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("Reranker JSON invalide : %s", text[:200])
                return {}
        else:
            return {}

    scores: dict[int, int] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id", -1))
                score = int(item.get("score", 0))
                if 0 <= idx < n:
                    scores[idx] = max(0, min(10, score))
            except (TypeError, ValueError):
                continue
    return scores


# ---------------------------------------------------------------------------
# Générateur de réponses
# ---------------------------------------------------------------------------


ANSWER_SYSTEM = """Tu es un assistant RAG pour une équipe technique (DevOps, PLM, infrastructure).
Tu réponds en français, de manière riche, précise et directe — sans politesses superflues.

Règles essentielles :
- Base ta réponse uniquement sur les passages fournis. Si l'info n'y est pas, dis-le.
- Cite tes sources avec [Source N].
- Pour les procédures, détaille TOUTES les étapes en texte (clics, menus, boutons).
  Lis les descriptions d'images pour extraire les noms de boutons/onglets précis.
- Préserve les commandes, codes, noms techniques exactement comme dans les sources.
- Sois exhaustif quand le contexte le permet : liste les bénéfices, les fonctions,
  les concepts. Ne te contente pas du minimum.
- Pour les questions "quels sont les X" / "types de Y" / "liste des Z", scanne
  TOUS les passages fournis pour extraire chaque élément mentionné. Si plusieurs
  passages parlent chacun d'un type/concept différent (ex: passage 1 = matière,
  passage 2 = semi-fini, passage 3 = produit fini), liste-les TOUS dans une
  même réponse consolidée. Ne pas se limiter aux éléments du premier passage.

Images (balises ![alt](rag-image://...)) :
- Si une image du contexte illustre ce que tu expliques, recopie sa balise EXACTEMENT
  telle quelle, juste après les étapes qu'elle illustre.
- Ne JAMAIS inventer de balise image. Si l'URL exacte rag-image://... n'est pas dans
  le contexte, n'écris pas de balise — le texte seul suffit.

Liens externes (URLs http:// ou https://) :
- Les passages peuvent contenir des URLs SharePoint, vidéos, présentations.
- Si une URL est pertinente pour la question, ajoute-la en fin de réponse dans une
  section "**Ressources complémentaires :**" sous forme `- [Libellé](URL_complète)`.
- Recopie l'URL ENTIÈRE avec ses paramètres (`?e=xxx` etc.) — ne pas raccourcir.
- Ne JAMAIS inclure de chemin local (`C:\\...`) comme ressource.
- Si aucune URL http(s):// pertinente n'est dans le contexte, omettre la section."""


ANSWER_PROMPT = """Contexte (passages extraits de la documentation interne) :

{context}

---

Question : {query}

Réponds en t'appuyant sur les passages ci-dessus. Cite les sources avec [Source N]."""


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict]          # [{num, title, source, ...}]
    used_chunks: list[RetrievedChunk]


class AnswerGenerator:
    """Génère une réponse finale à partir des chunks retrieved."""

    def __init__(self, client: GeminiClient, model: str):
        self.client = client
        self.model = model

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> RagAnswer:
        if not chunks:
            return RagAnswer(
                answer="Je n'ai trouvé aucun document pertinent dans la base pour répondre à cette question.",
                sources=[], used_chunks=[],
            )

        # Construire le contexte
        context_parts = []
        sources_info = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata or {}
            title = meta.get("doc_title") or meta.get("title") or meta.get("filename") or "document"
            source = meta.get("source") or meta.get("filepath") or meta.get("url") or ""
            doc_type = meta.get("doc_type") or meta.get("type") or "document"

            header = f"[Source {i}] ({doc_type}) {title}"
            if source:
                header += f" — {source}"
            context_parts.append(f"{header}\n{chunk.text}")
            sources_info.append({
                "num": i,
                "title": title,
                "source": source,
                "type": doc_type,
                "chunk_id": chunk.chunk_id,
            })

        context = "\n\n---\n\n".join(context_parts)
        prompt = ANSWER_PROMPT.format(context=context, query=query)

        answer = self.client.generate_text(
            self.model,
            prompt,
            system=ANSWER_SYSTEM,
            temperature=0.2,
            max_tokens=8192,
        )
        return RagAnswer(answer=answer.strip(), sources=sources_info, used_chunks=chunks)
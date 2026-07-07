"""Pipeline d'orchestration : ingestion + query."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from rag.chunker import MarkdownChunker
from rag.config import Config
from rag.converters import (
    ConvertedDoc, ConverterRegistry, LegacyOfficeConverter, MarkitdownConverter,
    MindManagerConverter, OfficeOpenXmlConverter, PdfConverter, VideoConverter,
    YouTubeConverter, iter_sources,
)
from rag.embeddings import GeminiEmbedder
from rag.images import ImageStore
from rag.llm import (
    AnswerGenerator, GeminiClient, GeminiReranker,
    make_video_transcriber, make_vision_describer, make_query_expander,
    RagAnswer,
)
from rag.rate_limit import QuotaExhaustedError, RateLimiter
from rag.retrieval import BM25Store, HybridRetriever
from rag.store import ChromaStore, DocEntry, DocState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap : instanciation des composants
# ---------------------------------------------------------------------------


@dataclass
class RagComponents:
    """Tous les composants nécessaires au RAG."""
    config: Config
    gemini: GeminiClient
    embedder: GeminiEmbedder
    chunker: MarkdownChunker
    converters: ConverterRegistry
    vector_store: ChromaStore
    bm25_store: BM25Store
    doc_state: DocState
    retriever: HybridRetriever
    reranker: GeminiReranker | None
    answerer: AnswerGenerator
    rate_limiter: RateLimiter
    image_store: ImageStore
    query_expander: callable | None = None


def build_components(config: Config) -> RagComponents:
    """Instancie tous les composants du RAG depuis la config."""
    # Rate limiter partagé entre tous les appels Gemini
    rate_limiter = RateLimiter(
        persist_path=config.storage_dir / "rate_state.json",
        limits_overrides=config.rate_limits_overrides,
    )

    gemini = GeminiClient(config.gemini_api_key, rate_limiter=rate_limiter)

    embedder = GeminiEmbedder(
        api_key=config.gemini_api_key,
        model=config.embedding_model,
        output_dim=config.embedding_dim,
        batch_size=config.embedding_batch_size,
        rate_limiter=rate_limiter,
    )

    chunker = MarkdownChunker(
        target_tokens=config.chunk_target_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
        min_chunk_chars=config.chunk_min_chars,
    )

    # Image store : commun à tous les converters qui extraient des images
    image_store = ImageStore(config.storage_dir / "images")

    # Vision describer : un seul, partagé entre converters (PDF, MindMap)
    vision_describer = make_vision_describer(gemini, config.vision_model)

    # Converters
    markitdown_conv = MarkitdownConverter()
    office_xml_conv = OfficeOpenXmlConverter(
        markitdown_converter=markitdown_conv,
        image_store=image_store if config.images_extract_office else None,
        vision_describer=vision_describer if config.images_extract_office else None,
        vision_cache_dir=config.image_cache_dir,
        enable_image_extraction=config.images_extract_office,
    )
    legacy_office_conv = LegacyOfficeConverter(
        markitdown_converter=markitdown_conv,
        cache_dir=config.storage_dir / "office_cache",
        # Le legacy converter pipe vers OfficeOpenXmlConverter après conversion COM
        # → les .doc/.ppt/.xls obtiennent aussi leurs images extraites
        office_xml_converter=office_xml_conv,
    )
    pdf_conv = PdfConverter(
        image_store=image_store if config.images_extract_pdf else None,
        vision_describer=vision_describer if config.images_extract_pdf else None,
        vision_cache_dir=config.image_cache_dir,
        enable_image_extraction=config.images_extract_pdf,
    )
    mindmap_conv = MindManagerConverter(
        vision_describer=vision_describer,
        cache_dir=config.image_cache_dir if config.mindmap_cache else None,
        enable_vision=config.mindmap_describe,
        image_store=image_store if config.images_extract_mindmap else None,
    )
    youtube_conv = YouTubeConverter(
        preferred_langs=config.youtube_langs,
        group_seconds=config.youtube_group_seconds,
    )
    video_transcriber = None
    if config.video_enable:
        video_transcriber = make_video_transcriber(
            gemini, config.video_model,
            processing_timeout=config.video_timeout,
        )
    video_conv = VideoConverter(
        transcriber=video_transcriber,
        cache_dir=config.video_cache_dir if config.video_cache else None,
        enable=config.video_enable,
    )
    # Ordre : spécifiques d'abord (mindmap, youtube, video, pdf, office_xml,
    # legacy_office), puis markitdown en fallback (html, txt, csv, etc.)
    converters = ConverterRegistry([
        mindmap_conv, youtube_conv, video_conv, pdf_conv,
        office_xml_conv, legacy_office_conv, markitdown_conv,
    ])

    vector_store = ChromaStore(config.chroma_dir, embedding_dim=config.embedding_dim)
    bm25_store = BM25Store(config.bm25_path)
    bm25_store.load()  # tente de charger l'index existant
    doc_state = DocState(config.doc_state_path)

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25_store,
        embedder=embedder,
        rrf_k=config.rrf_k,
    )

    reranker = None
    if config.use_reranker:
        reranker = GeminiReranker(gemini, config.reranker_model)

    answerer = AnswerGenerator(gemini, config.generation_model)

    # Query expander : utilise flash-lite (rapide et bon marché en quota)
    # Si l'utilisateur n'a pas configuré le multi-query, on le désactive.
    query_expander = None
    if getattr(config, "use_multi_query", True):
        # Le modèle d'expansion peut être configuré séparément, sinon on
        # réutilise le reranker_model qui est déjà flash-lite par défaut.
        expander_model = getattr(
            config, "query_expansion_model", config.reranker_model,
        )
        query_expander = make_query_expander(gemini, expander_model)

    return RagComponents(
        config=config, gemini=gemini, embedder=embedder, chunker=chunker,
        converters=converters, vector_store=vector_store, bm25_store=bm25_store,
        doc_state=doc_state, retriever=retriever, reranker=reranker,
        answerer=answerer, rate_limiter=rate_limiter, image_store=image_store,
        query_expander=query_expander,
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@dataclass
class IngestReport:
    total_sources: int = 0
    new_docs: int = 0
    updated_docs: int = 0
    unchanged_docs: int = 0
    deleted_docs: int = 0
    failed_sources: int = 0
    total_chunks: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def ingest(components: RagComponents, reset: bool = False,
           verbose: bool = False) -> IngestReport:
    """Ingère tous les documents du dossier data/.

    :param reset: si True, vide la base avant ingestion.
    :param verbose: si True, dump un preview des chunks pour inspection.
    """
    cfg = components.config
    report = IngestReport()

    if reset:
        logger.info("Reset de la base vectorielle demandé")
        components.vector_store.reset()
        components.doc_state = DocState(cfg.doc_state_path)
        # Supprimer le state file
        if cfg.doc_state_path.exists():
            cfg.doc_state_path.unlink()

    # 1. Scan des sources
    sources = list(iter_sources(cfg.data_dir))
    report.total_sources = len(sources)
    logger.info("Sources détectées : %d", len(sources))

    if not sources:
        logger.warning("Aucune source dans %s. Mets des fichiers dedans !", cfg.data_dir)
        return report

    # 2. Conversion + chunking + embedding pour chaque source
    seen_doc_ids: set[str] = set()
    preview_records: list[dict] = []

    for source in tqdm(sources, desc="Ingestion"):
        try:
            converted = components.converters.convert(source)
        except QuotaExhaustedError as exc:
            logger.error("Quota Gemini épuisé : %s", exc)
            report.errors.append(f"QUOTA: {exc}")
            print(f"\n⚠️  {exc}")
            print("    Ingestion interrompue. Relance-la plus tard (après minuit PT).")
            print("    Les docs déjà indexés sont préservés.")
            break
        except Exception as exc:
            logger.error("Échec conversion %s : %s", source, exc, exc_info=verbose)
            report.failed_sources += 1
            report.errors.append(f"{source}: {exc}")
            continue

        if converted is None:
            logger.debug("Skip %s (pas de converter ou contenu vide)", source)
            continue

        seen_doc_ids.add(converted.doc_id)

        # Check si déjà indexé avec le même contenu
        if components.doc_state.is_unchanged(converted.doc_id, converted.content_hash):
            report.unchanged_docs += 1
            logger.debug("Unchanged : %s", converted.title)
            continue

        # Nouveau ou modifié : virer les anciens chunks + images d'abord
        existing = components.doc_state.get(converted.doc_id)
        is_update = existing is not None
        if is_update:
            components.vector_store.delete(existing.chunk_ids)
            # Nettoyer aussi les anciennes images (elles seront re-extraites)
            components.image_store.remove_doc(converted.doc_id)

        # Chunker
        chunks = components.chunker.chunk(converted.markdown, doc_title=converted.title)
        if not chunks:
            logger.warning("Aucun chunk produit pour %s", converted.title)
            continue

        # Embed (peut être rate-limité ou bloqué si RPD épuisé)
        texts = [c.text for c in chunks]
        try:
            embeddings = components.embedder.embed_documents(texts)
        except QuotaExhaustedError as exc:
            logger.error("Quota Gemini épuisé à l'embedding : %s", exc)
            report.errors.append(f"QUOTA (embedding): {exc}")
            print(f"\n⚠️  {exc}")
            print("    Ingestion interrompue. Relance-la plus tard (après minuit PT).")
            print("    Les docs déjà indexés sont préservés.")
            break
        except Exception as exc:
            # Erreurs Gemini autres (400 token limit, 500, etc.) : on skip ce doc
            # mais on continue avec les autres au lieu de tout planter.
            err_msg = str(exc)
            if "exceeds the maximum number of tokens" in err_msg:
                logger.error(
                    "Doc trop gros (>1M tokens), skip : %s. "
                    "Réduis la taille du fichier source ou augmente l'agressivité du chunker.",
                    converted.title,
                )
                report.errors.append(
                    f"TOO_BIG: {converted.title} (>{1_000_000} tokens)"
                )
            else:
                logger.error("Erreur embedding sur %s : %s", converted.title, exc)
                report.errors.append(f"EMBED ERROR: {converted.title}: {exc}")
            report.failed_sources += 1
            continue

        # Upsert
        chunk_ids = [f"{converted.doc_id}_c{c.order:04d}" for c in chunks]
        metadatas = _build_chunk_metadatas(converted, chunks)

        components.vector_store.upsert(chunk_ids, texts, embeddings, metadatas)

        # Update state
        components.doc_state.set(DocEntry(
            doc_id=converted.doc_id,
            source=converted.source,
            title=converted.title,
            content_hash=converted.content_hash,
            chunk_ids=chunk_ids,
        ))

        if is_update:
            report.updated_docs += 1
        else:
            report.new_docs += 1
        report.total_chunks += len(chunks)

        # Preview en verbose
        if verbose:
            for c, cid in zip(chunks, chunk_ids):
                preview_records.append({
                    "chunk_id": cid,
                    "doc_title": converted.title,
                    "heading_path": c.heading_path,
                    "char_count": c.char_count,
                    "text_preview": c.text[:200],
                })

    # 3. Suppression des docs qui ont disparu du dossier data/
    existing_ids = {e.doc_id for e in components.doc_state.all()}
    missing = existing_ids - seen_doc_ids
    for doc_id in missing:
        entry = components.doc_state.get(doc_id)
        if entry:
            components.vector_store.delete(entry.chunk_ids)
            components.image_store.remove_doc(doc_id)
            components.doc_state.remove(doc_id)
            report.deleted_docs += 1
            logger.info("Supprimé (absent du dossier) : %s", entry.title)

    # 4. Persistence du doc_state
    components.doc_state.save()

    # 5. Reconstruction de l'index BM25 depuis tous les chunks actuels
    logger.info("Reconstruction de l'index BM25...")
    all_chunks = components.vector_store.get_all()
    components.bm25_store.build(all_chunks)

    # 6. Preview
    if verbose and preview_records:
        preview_path = cfg.storage_dir / "chunks_preview.jsonl"
        with open(preview_path, "w", encoding="utf-8") as f:
            for rec in preview_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Preview des chunks écrit dans %s", preview_path)

    return report


def _build_chunk_metadatas(doc: ConvertedDoc, chunks: list) -> list[dict]:
    """Construit les metadata pour chaque chunk."""
    metas = []
    for c in chunks:
        meta = {
            "doc_id": doc.doc_id,
            "doc_title": doc.title,
            "source": doc.source,
            "doc_type": doc.metadata.get("type", ""),
            "chunk_order": c.order,
            "heading_path": " > ".join(c.heading_path) if c.heading_path else "",
        }
        # Copier les autres metadata (url, filepath, etc.)
        for k, v in doc.metadata.items():
            if k not in meta:
                meta[k] = v
        metas.append(meta)
    return metas


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _diversify_candidates(candidates, max_per_doc: int = 4):
    """Diversifie les candidats par document via round-robin.

    Évite qu'un gros document (ex: mindmap CREO avec 47 chunks) ne monopolise
    les top candidats au détriment d'autres docs pertinents (ex: un PDF qui
    contient une bonne définition).

    Algorithme :
    1. Grouper les candidats par doc_id (en préservant l'ordre dans chaque groupe)
    2. Faire un round-robin pour intercaler les docs (rang 1 du doc A, rang 1
       du doc B, rang 2 du doc A, rang 2 du doc B...)
    3. Limiter à max_per_doc par doc dans le résultat principal, le reste va
       en overflow.

    Cela garantit que tous les docs représentés ont leurs meilleurs chunks
    visibles avant le rerank, même si un seul doc domine en score brut.
    """
    if not candidates:
        return candidates

    # Grouper par doc_id en préservant l'ordre
    groups: dict[str, list] = {}
    doc_order: list[str] = []  # ordre d'apparition des docs
    for c in candidates:
        doc_id = c.metadata.get("doc_id", "")
        if doc_id not in groups:
            groups[doc_id] = []
            doc_order.append(doc_id)
        groups[doc_id].append(c)

    # Round-robin : rang 1 de chaque doc, puis rang 2 de chaque doc, etc.
    primary: list = []
    overflow: list = []
    max_rank = max(len(g) for g in groups.values())
    for rank in range(max_rank):
        for doc_id in doc_order:
            group = groups[doc_id]
            if rank < len(group):
                if rank < max_per_doc:
                    primary.append(group[rank])
                else:
                    overflow.append(group[rank])

    return primary + overflow


def query(components: RagComponents, question: str,
          verbose: bool = False) -> RagAnswer:
    """Exécute le pipeline de query : retrieve + rerank + generate."""
    cfg = components.config

    # 1. Retrieval hybride (avec multi-query si activé)
    queries = [question]
    if components.query_expander is not None:
        try:
            queries = components.query_expander(question)
            if verbose and len(queries) > 1:
                logger.info("Query expansion : %d queries dérivées", len(queries))
                for i, q in enumerate(queries):
                    logger.info("  [%d] %s", i, q)
        except Exception as exc:
            logger.warning("Query expansion erreur, fallback single : %s", exc)
            queries = [question]

    try:
        if len(queries) > 1 and hasattr(components.retriever, "multi_search"):
            candidates = components.retriever.multi_search(queries, k=cfg.rerank_k)
        else:
            candidates = components.retriever.search(question, k=cfg.rerank_k)
    except QuotaExhaustedError as exc:
        return RagAnswer(
            answer=f"⚠️ Quota Gemini épuisé pour aujourd'hui : {exc}\n\n"
                   f"Impossible d'embedder la question. Réessaie après minuit Pacific Time "
                   f"(ou passe en Tier 1 dans Google AI Studio).",
            sources=[], used_chunks=[],
        )

    # Diversification : limiter les candidats par document pour éviter qu'un
    # seul gros doc (ex: une mindmap avec 47 chunks) ne sature le top des
    # candidats et écrase les autres docs pertinents (ex: un PDF qui a aussi
    # la réponse). On garde max N chunks par doc avant rerank.
    candidates = _diversify_candidates(candidates, max_per_doc=4)

    if verbose:
        logger.info("%d candidats avant rerank", len(candidates))
        for i, c in enumerate(candidates[:10]):
            logger.info(
                "  [%d] fused=%.4f vec=%s bm25=%s — %s",
                i,
                c.fused_score,
                f"{c.vector_score:.3f}" if c.vector_score is not None else "—",
                f"{c.bm25_score:.3f}" if c.bm25_score is not None else "—",
                c.metadata.get("doc_title", "?")[:60],
            )

    # 2. Rerank
    if components.reranker and candidates:
        final = components.reranker.rerank(question, candidates, top_k=cfg.top_k)
    else:
        final = sorted(candidates, key=lambda c: c.fused_score, reverse=True)[:cfg.top_k]

    if verbose:
        logger.info("%d chunks après rerank (top_k=%d)", len(final), cfg.top_k)
        for i, c in enumerate(final):
            logger.info("  [%d] %s", i + 1, c.metadata.get("doc_title", "?")[:80])

    # 3. Génération
    answer = components.answerer.generate(question, final)
    return answer


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def print_stats(components: RagComponents):
    """Affiche un résumé de la base."""
    print(f"\n=== État de la base ===")
    print(f"Documents indexés : {len(components.doc_state.all())}")
    print(f"Chunks totaux     : {components.vector_store.count()}")
    print(f"Config embeddings : {components.config.embedding_model} "
          f"({components.config.embedding_dim} dims)")
    print(f"Chroma path       : {components.config.chroma_dir}")
    print()

    if not components.doc_state.all():
        print("(aucun document)\n")
        return

    print(f"{'Type':<12} {'Chunks':>7}  {'Titre'}")
    print("-" * 80)
    for entry in sorted(components.doc_state.all(), key=lambda e: e.title):
        # Récupérer le type depuis les chunks
        doc_type = "?"
        if entry.chunk_ids:
            try:
                result = components.vector_store._collection.get(
                    ids=[entry.chunk_ids[0]], include=["metadatas"],
                )
                if result["metadatas"]:
                    doc_type = result["metadatas"][0].get("doc_type", "?")
            except Exception:
                pass
        title = entry.title[:55] + "..." if len(entry.title) > 58 else entry.title
        print(f"{doc_type:<12} {len(entry.chunk_ids):>7}  {title}")
    print()
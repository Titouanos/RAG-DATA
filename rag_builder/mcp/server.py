"""Serveur MCP : `rag_query(collection, question)` et outils d'inventaire.

Client de `RagService` au même titre que l'API. En mode Qdrant local, à lancer seul (accès
exclusif à storage/qdrant) ; en mode serveur (Docker), peut cohabiter avec l'API.

Transports : `stdio` (défaut, clients type Claude Desktop) ou `streamable-http`.
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Any

from rag_builder.config import get_settings
from rag_builder.core.images import INTERNAL_SCHEME
from rag_builder.core.rag_service import RagService
from rag_builder.logging_conf import setup_logging

logger = logging.getLogger(__name__)

# Docstring-prompt empirique (portée du POC) : instructions pour le LLM appelant.
RAG_QUERY_DOC = """Interroge une base documentaire interne et renvoie une réponse sourcée.

Comportement attendu de ta réponse :
- Réponds UNIQUEMENT à partir des extraits fournis ; si l'information n'y est pas, dis-le.
- DÉTAILLE les procédures étape par étape ; pour les questions énumératives, recense tous
  les éléments présents dans les extraits.
- Cite les sources avec [n] (voir le champ "sources").
- Recopie EXACTEMENT les éventuelles balises d'image ![...](...) présentes dans "answer",
  sans en inventer.

Args:
    collection: nom de la collection à interroger.
    question: la question de l'utilisateur, recopiée telle quelle.

Returns:
    {answer: str (markdown, avec [n]), sources: [{n, source_name, page_or_section, score}]}
"""


def _build_service(settings) -> RagService:
    if settings.app_db_path.exists():
        from rag_builder.db.session import make_engine
        from rag_builder.db.sql_registry import SqlCollectionRegistry

        registry = SqlCollectionRegistry(make_engine(settings.app_db_path))
        return RagService.from_settings(settings, registry=registry)
    return RagService.from_settings(settings)


def build_mcp(service: RagService, image_base_url: str | None = None):
    """Construit le serveur FastMCP branché sur `service`."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("rag-builder")
    ref_re = re.compile(r"!\[([^\]]*)\]\(" + re.escape(INTERNAL_SCHEME) + r"([^)]+)\)")

    def rewrite_images(markdown: str, collection: str) -> str:
        if not image_base_url:
            return markdown
        base = image_base_url.rstrip("/")

        def repl(m: re.Match) -> str:
            rel = m.group(2)  # <collection>/<doc>/<file>
            parts = rel.split("/")
            return f"![{m.group(1)}]({base}/collections/{collection}/images/{'/'.join(parts[1:])})"

        return ref_re.sub(repl, markdown)

    @mcp.tool(description=RAG_QUERY_DOC)
    def rag_query(collection: str, question: str) -> dict[str, Any]:
        try:
            result = service.retrieve(collection, question)
            answer = "".join(service.stream_answer(collection, question, result))
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "answer": "", "sources": []}
        return {
            "answer": rewrite_images(answer, collection),
            "sources": [
                {
                    "n": s["n"],
                    "source_name": s["source_name"],
                    "page_or_section": s["page_or_section"],
                    "score": s["score"],
                }
                for s in result.sources()
            ],
        }

    @mcp.tool()
    def list_collections() -> list[dict[str, Any]]:
        """Liste les collections disponibles (nom + description)."""
        return [{"name": m.name, "description": m.description} for m in service.registry.list()]

    @mcp.tool()
    def rag_stats(collection: str) -> dict[str, Any]:
        """État d'une collection : nb de documents, de chunks, modèle d'embedding."""
        meta = service.registry.get(collection)
        if meta is None:
            return {"error": f"collection introuvable : {collection}"}
        return {
            "documents": len(service.store.list_doc_ids(collection)),
            "chunks": service.store.count(collection),
            "embedding_model": meta.embedding_model,
        }

    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serveur MCP RAG Builder")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--image-base-url", default=None,
                        help="URL de base de l'API pour réécrire les images (ex. http://host:8000)")
    args = parser.parse_args(argv)

    # Logs sur stderr : stdout est réservé au transport stdio.
    setup_logging()

    settings = get_settings()
    service = _build_service(settings)
    mcp = build_mcp(service, image_base_url=args.image_base_url)
    if args.transport == "streamable-http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
    logger.info("Serveur MCP RAG Builder (transport=%s)", args.transport)
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

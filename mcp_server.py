"""Serveur MCP qui expose le RAG à OpenWebUI (et autres clients MCP).

Architecture :
- **MCP server** (stdio) : expose 2 outils que les clients MCP appellent.
- **HTTP statique** (FastAPI) : sert les images au navigateur d'OpenWebUI.

Le serveur réécrit les références internes `rag-image://doc_id/img.png` présentes
dans les réponses RAG en URLs HTTP absolues vers son propre endpoint /images.

Outils MCP exposés :
- `rag_query(question)` : pose une question au RAG, retourne markdown + sources.
- `rag_stats()` : retourne l'état de la base (nb docs, nb chunks).

Pour brancher OpenWebUI :

    # Étape 1 — démarrer le serveur HTTP des images (à laisser tourner dans un terminal)
    python mcp_server.py --http-only

    # Étape 2 — exposer le RAG en API OpenAPI via mcpo (autre terminal)
    pip install mcpo
    mcpo --port 8000 -- python mcp_server.py --no-http

    # Étape 3 — dans OpenWebUI : Settings → Tools → Add OpenAPI Tool
    #          URL: http://localhost:8000
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Ajout du répertoire parent au sys.path
sys.path.insert(0, str(Path(__file__).parent))

from rag.config import load_config
from rag.images import INTERNAL_SCHEME
from rag.pipeline import build_components, query as rag_pipeline_query, RagComponents

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Globals (initialisés au boot, partagés entre MCP et HTTP)
# ---------------------------------------------------------------------------

_components: RagComponents | None = None
_image_base_url: str = "http://localhost:8765/images/"


def _set_image_base_url(host: str, port: int):
    global _image_base_url
    # On utilise toujours localhost dans le scheme pour éviter les problèmes
    # de réseau en local. À adapter si tu déploies sur un serveur distant.
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    _image_base_url = f"http://{display_host}:{port}/images/"


# ---------------------------------------------------------------------------
# Réécriture rag-image:// → http://.../images/...
# ---------------------------------------------------------------------------


_INTERNAL_REF_RE = re.compile(
    r"!\[([^\]]*)\]\(" + re.escape(INTERNAL_SCHEME) + r"([^)]+)\)"
)


def rewrite_image_references(markdown: str) -> str:
    """Remplace les rag-image://... par des URLs HTTP absolues.

    Exemple :
        ![desc](rag-image://doc_abc/img_001.png)
        →
        ![desc](http://localhost:8765/images/doc_abc/img_001.png)
    """
    def _replace(m):
        alt = m.group(1)
        relative = m.group(2)
        return f"![{alt}]({_image_base_url}{relative})"
    return _INTERNAL_REF_RE.sub(_replace, markdown)


# ---------------------------------------------------------------------------
# Serveur HTTP statique pour les images (FastAPI)
# ---------------------------------------------------------------------------


def make_http_app():
    """Construit l'application FastAPI qui sert les images."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="RAG Image Server", version="0.1.0")

    # CORS large : OpenWebUI tournera probablement sur un autre port/host
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "rag_ready": _components is not None}

    @app.get("/images/{full_path:path}")
    def serve_image(full_path: str):
        if _components is None:
            raise HTTPException(status_code=503, detail="RAG not initialized")
        resolved = _components.image_store.resolve(full_path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Image not found")
        ext = resolved.suffix.lower()
        media_type = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
        }.get(ext, "application/octet-stream")
        return FileResponse(resolved, media_type=media_type)

    return app


def run_http_server(host: str, port: int):
    """Lance le serveur HTTP en mode bloquant."""
    import uvicorn
    app = make_http_app()
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Serveur MCP
# ---------------------------------------------------------------------------


def make_mcp_server():
    """Construit le serveur MCP avec FastMCP."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("rag-poc")

    @mcp.tool()
    def rag_query(question: str) -> dict[str, Any]:
        """Pose une question à la base documentaire interne (Teamcenter, PLM, infrastructure, tutoriels métier).

        Comportement attendu pour ta réponse :

        - DÉTAILLE LES ÉTAPES en texte pour les procédures. Lis les descriptions
          d'images (alt) pour extraire les noms de boutons/menus exacts.
        - SOIS RICHE : si le contexte liste des bénéfices, fonctions, concepts,
          recopie-les. Ne te contente pas du minimum.
        - QUESTIONS ÉNUMÉRATIVES ("quels sont les X", "types de Y", "liste des Z") :
          scanne TOUS les passages pour extraire chaque élément. Si chaque passage
          décrit un élément différent, consolide-les dans une même liste.
        - PRÉSERVE LA QUESTION sans la reformuler.

        Images du champ "answer" :
        - Recopie EXACTEMENT les balises ![...](http://localhost:8765/images/...)
          ou ![...](rag-image://...) déjà présentes, sans HTML-escape.
        - Ne JAMAIS en inventer : si l'URL exacte n'est pas dans answer, n'écris
          pas de balise — le texte seul suffit.

        Liens externes (URLs http:// ou https://) du champ "answer" :
        - Si pertinents, ajoute-les en fin de réponse dans "**Ressources
          complémentaires :**" sous forme `- [Libellé](URL_complète)`.
        - Recopie l'URL ENTIÈRE avec ses paramètres (`?e=xxx`) — ne pas
          uppercase, raccourcir, ou nettoyer.
        - JAMAIS de chemin local (`C:\\...`) comme ressource.
        - Si aucune URL http(s) pertinente, omettre la section.

        Args:
            question: La question utilisateur, recopiée telle quelle.

        Returns:
            Dict avec :
            - answer: la réponse markdown (avec balises d'images si pertinent)
            - sources: liste des documents cités, avec leur titre et chemin
        """
        if _components is None:
            return {"error": "RAG not initialized", "answer": "", "sources": []}

        try:
            result = rag_pipeline_query(_components, question)
        except Exception as exc:
            logger.error("rag_query a échoué : %s", exc, exc_info=True)
            return {"error": str(exc), "answer": "", "sources": []}

        # Réécriture des références d'images en URLs HTTP servies
        answer_md = rewrite_image_references(result.answer)

        sources_simplified = [
            {
                "num": s["num"],
                "title": s["title"],
                "type": s["type"],
                "source": s["source"],
            }
            for s in result.sources
        ]

        return {
            "answer": answer_md,
            "sources": sources_simplified,
        }

    @mcp.tool()
    def rag_stats() -> dict[str, Any]:
        """Retourne l'état de la base RAG : nombre de documents indexés et de chunks.

        Returns:
            Dict avec :
            - documents_indexed: nombre de documents
            - chunks_indexed: nombre de chunks vectorisés
            - embedding_model: nom du modèle d'embedding utilisé
            - embedding_dim: dimension des vecteurs
        """
        if _components is None:
            return {"error": "RAG not initialized"}

        return {
            "documents_indexed": len(_components.doc_state.all()),
            "chunks_indexed": _components.vector_store.count(),
            "embedding_model": _components.config.embedding_model,
            "embedding_dim": _components.config.embedding_dim,
        }

    return mcp


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Serveur MCP du RAG")
    parser.add_argument("--http-host", default="127.0.0.1",
                        help="Host pour le serveur HTTP des images")
    parser.add_argument("--http-port", type=int, default=8765,
                        help="Port pour le serveur HTTP des images")
    parser.add_argument("--no-http", action="store_true",
                        help="Désactive le serveur HTTP intégré (à utiliser quand "
                             "un autre process sert déjà les images)")
    parser.add_argument("--http-only", action="store_true",
                        help="Lance uniquement le serveur HTTP (pas de MCP). "
                             "Utile pour avoir un process dédié aux images.")
    parser.add_argument("--config", default="config.yaml",
                        help="Chemin du fichier de config")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Logs détaillés")
    args = parser.parse_args()

    # Logs : sur stderr seulement (stdio MCP utilise stdout)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ["httpx", "chromadb", "urllib3", "uvicorn.access"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 1. Charger la config et instancier le RAG
    try:
        config = load_config(args.config)
    except RuntimeError as exc:
        print(f"ERREUR CONFIG : {exc}", file=sys.stderr)
        sys.exit(1)

    global _components
    logger.info("Initialisation du RAG...")
    _components = build_components(config)
    logger.info(
        "RAG prêt : %d docs, %d chunks",
        len(_components.doc_state.all()),
        _components.vector_store.count(),
    )

    # 2. Configurer l'URL d'images si le serveur HTTP va tourner
    if not args.no_http:
        _set_image_base_url(args.http_host, args.http_port)

    # 3. Mode http-only : juste le serveur HTTP, pas de MCP
    if args.http_only:
        logger.info("Mode HTTP-only sur http://%s:%d", args.http_host, args.http_port)
        run_http_server(args.http_host, args.http_port)
        return

    # 4. Lancer le serveur HTTP statique pour les images dans un thread
    if not args.no_http:
        logger.info("Serveur HTTP images sur http://%s:%d", args.http_host, args.http_port)
        http_thread = threading.Thread(
            target=run_http_server, args=(args.http_host, args.http_port),
            daemon=True,
        )
        http_thread.start()
        # Petit sleep pour que le serveur HTTP soit prêt avant le MCP
        time.sleep(0.5)

    # 5. Lancer le serveur MCP en stdio (mode bloquant)
    mcp = make_mcp_server()
    logger.info("Serveur MCP démarré (transport=stdio)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
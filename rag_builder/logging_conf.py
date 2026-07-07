"""Configuration centralisée du logging (remplace les 3 blocs dupliqués du POC)."""

from __future__ import annotations

import logging
import sys

# Loggers tiers trop verbeux qu'on abaisse à WARNING.
_NOISY = ["httpx", "httpcore", "urllib3", "qdrant_client", "sentence_transformers",
          "transformers", "FlagEmbedding", "uvicorn.access", "filelock", "fsspec"]


def setup_logging(level: int = logging.INFO, *, stream=None) -> None:
    """Initialise le logging racine.

    :param level: niveau du logger racine (DEBUG en mode verbeux).
    :param stream: flux de sortie ; par défaut stderr (impératif quand un transport
        stdio comme MCP occupe stdout).
    """
    logging.basicConfig(
        level=level,
        stream=stream or sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in _NOISY:
        logging.getLogger(noisy).setLevel(logging.WARNING)

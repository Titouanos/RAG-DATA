"""Retry avec backoff pour l'établissement d'un flux de génération.

Le streaming ne se rejoue pas en cours de route : on ne réessaie que **l'ouverture** du
flux et l'obtention du **premier token** (là où se concentrent les erreurs transitoires :
réseau, 429, 5xx). Une fois le premier token émis, les erreurs sont propagées.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

_SENTINEL = object()


def stream_with_retries(
    factory: Callable[[], Iterator[str]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    is_retriable: Callable[[Exception], bool] = lambda _e: True,
) -> Iterator[str]:
    """Ouvre le flux via `factory` avec retries sur erreurs transitoires (avant 1er token)."""
    for attempt in range(1, attempts + 1):
        try:
            it = factory()
            first = next(it, _SENTINEL)
        except Exception as exc:  # noqa: BLE001
            if attempt >= attempts or not is_retriable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Génération : tentative %d/%d échouée (%s), retry dans %.1fs",
                           attempt, attempts, exc, delay)
            time.sleep(delay)
            continue
        # Flux ouvert : on émet le premier token puis le reste sans retry.
        if first is not _SENTINEL:
            yield first  # type: ignore[misc]
        yield from it
        return

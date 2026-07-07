"""Utilitaires de retrieval.

La fusion **dense + sparse intra-requête** est déléguée à Qdrant (prefetch + FusionQuery
RRF, cf. `store.py`). `rrf_fuse` reste disponible pour la fusion **inter-requêtes** de la
query expansion (option désactivée par défaut, Phase 4) et pour les tests.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def rrf_fuse(*ranked_lists: Sequence[str], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion sur N listes d'IDs ordonnées par pertinence.

    Formule : score(id) = Σ_listes 1/(k + rang) avec rang 0-based. Poids égaux entre
    listes. Retourne [(id, score)] trié par score décroissant.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

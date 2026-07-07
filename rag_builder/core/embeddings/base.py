"""Interface `Embedder` : abstraction du modèle d'embedding d'une collection.

Le modèle est **figé à la création de la collection** (stocké en métadonnée) : en
changer impose une réindexation complète. Chaque implémentation expose son `model_id`
(vérifié à l'ouverture d'une collection) et sa `dense_dim`.

L'asymétrie `embed_documents` / `embed_query` est conservée : certains modèles (Gemini)
encodent différemment documents et requêtes. bge-m3 fournit en plus un vecteur **sparse**
lexical (`supports_sparse=True`) exploité par le stockage hybride Qdrant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag_builder.core.models import SparseVector


@dataclass
class Embedding:
    """Vecteurs produits pour un texte : dense (toujours) + sparse (optionnel)."""

    dense: list[float]
    sparse: SparseVector | None = None


class Embedder(ABC):
    """Contrat commun à toutes les implémentations d'embedding."""

    #: Identifiant du modèle, stocké en métadonnée de collection (ex. "BAAI/bge-m3").
    model_id: str
    #: Dimension du vecteur dense.
    dense_dim: int
    #: True si l'embedder produit aussi un vecteur sparse lexical.
    supports_sparse: bool = False

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        """Encode une liste de chunks pour indexation."""

    @abstractmethod
    def embed_query(self, text: str) -> Embedding:
        """Encode une requête utilisateur."""

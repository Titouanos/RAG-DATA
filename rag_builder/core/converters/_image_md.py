"""Helpers markdown pour les images décrites par Vision (partagés PDF/mindmap).

Ces fonctions produisent des blocs markdown propres à partir des descriptions
Vision (souvent structurées en markdown) et filtrent les images décoratives. Elles
sont internes au package `converters` et ne dépendent d'aucun provider.
"""

from __future__ import annotations

import re

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def mime_for_ext(ext: str) -> str:
    """Type MIME correspondant à une extension d'image (défaut image/png)."""
    return _MIME_BY_EXT.get(ext.lower(), "image/png")


def sanitize_alt_text(text: str) -> str:
    """Nettoie un alt text d'image markdown pour éviter les casses de rendu.

    Aplati les sauts de ligne, retire le markdown imbriqué (gras, italique,
    titres, code, puces), neutralise les crochets, remplace les guillemets
    droits par des typographiques et cape la longueur à 150 caractères.
    """
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **gras**
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italique* / puces de liste
    text = re.sub(r"#+\s*", "", text)  # # titres
    text = re.sub(r"`+", "", text)  # `code`
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace(chr(0x27), chr(0x2019))  # apostrophe droite -> typographique
    text = text.replace(chr(0x22), chr(0xAB))  # guillemet droit -> «
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 150:
        cut = text[:147]
        last_space = cut.rfind(" ")
        if last_space > 100:
            cut = cut[:last_space]
        text = cut + "..."
    return text


def build_image_block(description: str, image_reference: str) -> str:
    """Bloc markdown combinant une description riche (indexée) + une balise image.

    Format::

        *Description complète riche en mots-clés...*

        ![alt court](rag-image://...)

    La description complète (italique) est indexée pour le retrieval ; l'alt court
    (~80 caractères) permet au LLM de recopier la balise sans se perdre.
    """
    if not description:
        return ""

    rich_desc = description.replace("\n", " ").replace("\r", " ")
    rich_desc = re.sub(r"\*\*([^*]+)\*\*", r"\1", rich_desc)
    rich_desc = re.sub(r"\*([^*]+)\*", r"\1", rich_desc)
    rich_desc = re.sub(r"#+\s*", "", rich_desc)
    rich_desc = re.sub(r"`+", "", rich_desc)
    rich_desc = re.sub(r"\s+", " ", rich_desc).strip()

    alt_short = sanitize_alt_text(description)
    if len(alt_short) > 80:
        cut = alt_short[:77]
        last_space = cut.rfind(" ")
        if last_space > 50:
            cut = cut[:last_space]
        alt_short = cut + "..."

    return f"*{rich_desc}*\n\n![{alt_short}]({image_reference})"


# Patterns révélant qu'une image est décorative (logo, slide de titre, bandeau,
# pictogramme abstrait...). Identifiés à partir de vraies descriptions Vision.
_DECORATIVE_PATTERNS = (
    "il n'y a pas d'interface logicielle visible",
    "aucune action spécifique n'est illustrée",
    "aucune action spécifique (clic",
    "il s'agit d'un visuel statique",
    "l'image est purement illustrative",
    "l'image semble être une simple représentation graphique",
    "l'image est une simple représentation graphique",
    "purement illustrative et ne montre aucune action",
    "ne montre aucune action spécifique",
    "montre le logo de",
    "présente le logo de",
    "présente le logotype",
    "image de marque",
    "présente un visuel promotionnel",
    "visuel promotionnel ou informatif",
    "visuel statique",
    "page de remerciement",
    "page de conclusion",
    "diapositive de remerciement",
    "diapositive de conclusion",
    "diapositive de titre",
    "diapositive d'introduction",
    "slide de titre",
    "page de garde",
    "page de couverture",
    "présente une diapositive de présentation intitulée",
    "illustration vectorielle représentant",
    "illustration vectorielle stylisée",
    "scène industrielle stylisée",
    "fond décoratif",
    "présente une vignette graphique",
    "aucun logiciel, interface",
    "il n'y a aucun texte lisible",
    "aucun texte n'est visible",
    "aucun texte lisible",
    "aucun élément textuel",
    "ne contient aucun texte",
    "il s'agit d'un élément graphique",
    "élément graphique abstrait",
    "élément décoratif",
    "forme géométrique",
    "trait diagonal",
    "barre oblique",
    "ligne diagonale verte",
    "présente un fond blanc",
    "image quasi vide",
    "image principalement blanche",
    "image presque entièrement blanche",
    "espace vide avec",
    "fond blanc avec une simple",
    "fond blanc avec un",
    "fond blanc traversé",
    "un simple trait",
    "une simple forme",
)


def is_decorative_image(description: str) -> bool:
    """Détecte si la description Vision suggère une image décorative (à ignorer)."""
    if not description:
        return False
    desc_lower = description.lower()
    return any(p in desc_lower for p in _DECORATIVE_PATTERNS)

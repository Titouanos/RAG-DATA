"""Chunking markdown-aware.

Stratégie :
1. Découpage par hiérarchie de titres (H1..H6) — chaque section devient un bloc.
2. Si un bloc dépasse la taille cible, on le resplit sur les paragraphes
   avec overlap pour garder la continuité.
3. Chaque chunk conserve le **chemin des titres parents** en préfixe.
   Exemple : "Guide DevOps > Jenkins > Erreurs\n\nQuand le pipeline..."
   → le LLM comprend le contexte sans avoir besoin des chunks voisins.
4. Les chunks trop courts (< min_chars) sont fusionnés avec le suivant pour
   éviter le bruit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Chunk:
    """Unité d'indexation."""

    text: str                           # texte final à embedder et indexer
    heading_path: List[str] = field(default_factory=list)  # ex: ["Guide", "Section"]
    order: int = 0                      # position dans le doc
    char_count: int = 0
    metadata: dict = field(default_factory=dict)


class MarkdownChunker:
    """Chunker qui respecte la structure markdown."""

    def __init__(self, target_tokens: int = 500, overlap_tokens: int = 60,
                 min_chunk_chars: int = 100):
        # Approximation char <-> token : en français ~4 chars par token
        self.target_chars = target_tokens * 4
        self.overlap_chars = overlap_tokens * 4
        self.min_chunk_chars = min_chunk_chars

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def chunk(self, markdown: str, doc_title: str | None = None) -> List[Chunk]:
        """Chunke un document markdown."""
        if not markdown or not markdown.strip():
            return []

        sections = self._split_by_headings(markdown)

        chunks: List[Chunk] = []
        order = 0
        for heading_path, body in sections:
            body = body.strip()
            if not body:
                continue

            # Contexte : le chemin des titres + éventuellement titre du doc
            context_path = []
            if doc_title:
                context_path.append(doc_title)
            context_path.extend(heading_path)

            for piece in self._split_by_size(body):
                piece = piece.strip()
                if not piece:
                    continue
                header_prefix = " > ".join(context_path)
                if header_prefix:
                    text = f"[{header_prefix}]\n\n{piece}"
                else:
                    text = piece
                chunks.append(Chunk(
                    text=text,
                    heading_path=list(heading_path),
                    order=order,
                    char_count=len(text),
                ))
                order += 1

        # Fusion des chunks trop courts avec le suivant
        chunks = self._merge_tiny_chunks(chunks)
        return chunks

    # ------------------------------------------------------------------
    # Découpage par titres
    # ------------------------------------------------------------------

    def _split_by_headings(self, md: str) -> List[tuple[List[str], str]]:
        """Retourne [(heading_path, body), ...]."""
        lines = md.split("\n")
        sections: List[tuple[List[str], str]] = []
        stack: List[tuple[int, str]] = []  # [(level, title), ...]
        buffer: List[str] = []
        current_path: List[str] = []

        def flush():
            if buffer:
                body = "\n".join(buffer).strip()
                if body:
                    sections.append((list(current_path), body))

        for line in lines:
            m = HEADING_RE.match(line)
            if m:
                flush()
                buffer.clear()
                level = len(m.group(1))
                title = m.group(2).strip()
                # Pop les titres de niveau >= au nouveau
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current_path = [t for _, t in stack]
            else:
                buffer.append(line)
        flush()

        # Si aucun titre trouvé, tout le doc est une section unique
        if not sections:
            body = md.strip()
            if body:
                sections.append(([], body))

        return sections

    # ------------------------------------------------------------------
    # Redécoupage si trop gros
    # ------------------------------------------------------------------

    def _split_by_size(self, text: str) -> List[str]:
        """Si le texte dépasse target_chars, le découpe sur les paragraphes."""
        if len(text) <= self.target_chars:
            return [text]

        # Split par paragraphes (double newline)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Si le paragraphe tout seul est énorme, split brutal
            if len(para) > self.target_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._hard_split(para))
                continue

            # Test : ajout au chunk courant ?
            candidate = f"{current}\n\n{para}" if current else para
            if len(candidate) <= self.target_chars:
                current = candidate
            else:
                # Flush avec overlap
                chunks.append(current.strip())
                overlap = self._extract_overlap(current)
                current = f"{overlap}\n\n{para}" if overlap else para

        if current.strip():
            chunks.append(current.strip())

        # Réparer les URLs coupées : si la frontière entre deux chunks tombe
        # au milieu d'une URL http(s)://..., déplacer la frontière pour garder
        # l'URL intacte dans un seul chunk.
        chunks = self._repair_split_urls(chunks)

        return chunks

    @staticmethod
    def _repair_split_urls(chunks: List[str]) -> List[str]:
        """Évite de couper une URL en deux entre chunks consécutifs.

        Cas typique : un chunk se termine par 'https://gahub.sharepoint.com/'
        et le suivant commence par ':v:/s/IAO/...?e=xxx)'. Le LLM verrait deux
        bouts d'URL et ne pourrait pas la reconstruire.

        Stratégie : pour chaque paire (chunk_i, chunk_i+1), si chunk_i contient
        une URL en fin ET chunk_i+1 commence par un caractère qui prolonge
        clairement une URL (slash, ?, &, =, paramètres, etc.), alors déplacer
        la fin du chunk_i vers le début du chunk_i+1.

        On ne touche PAS si le chunk suivant commence par une lettre/chiffre,
        car ça peut être un nouveau paragraphe (ex: "https://example.com/page1.\\n\\nEt
        ensuite..."). Trop ambigu.
        """
        if len(chunks) < 2:
            return chunks

        # URL se terminant le chunk courant
        url_re = re.compile(r"https?://[^\s\)\]\"'<>]+$")
        # Caractères qui prolongent VRAIMENT une URL (non ambigus)
        url_continuation_chars = set("/:?&=#%+~,;_-")

        repaired: List[str] = []
        i = 0
        chunks = list(chunks)

        while i < len(chunks):
            current = chunks[i]
            if i + 1 < len(chunks):
                m = url_re.search(current)
                next_chunk = chunks[i + 1]
                if m and next_chunk and next_chunk[0] in url_continuation_chars:
                    # URL coupée : déplacer la fin de current vers le début de next
                    url_start = m.start()
                    truncated_url = current[url_start:]
                    current = current[:url_start].rstrip()
                    chunks[i + 1] = f"{truncated_url}{next_chunk}"
            if current.strip():
                repaired.append(current)
            i += 1
        return repaired

    def _hard_split(self, text: str) -> List[str]:
        """Split brutal d'un texte > target_chars (essaie les phrases)."""
        # Tente de split sur les phrases
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        for sent in sentences:
            candidate = f"{current} {sent}" if current else sent
            if len(candidate) <= self.target_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(sent) > self.target_chars:
                    # phrase géante, split par chars avec overlap
                    step = self.target_chars - self.overlap_chars
                    for i in range(0, len(sent), step):
                        chunks.append(sent[i:i + self.target_chars])
                    current = ""
                else:
                    current = sent
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _extract_overlap(self, text: str) -> str:
        """Prend les derniers overlap_chars du texte, au niveau d'une phrase.

        Évite de couper au milieu d'une référence d'image markdown : si l'overlap
        tombe dans `![alt](url)`, on coupe avant le `!`.
        """
        if self.overlap_chars <= 0 or not text:
            return ""
        tail = text[-self.overlap_chars * 2:]  # un peu plus pour avoir du jeu
        # Cherche la première frontière de phrase dans la seconde moitié
        half = len(tail) // 2
        matches = list(re.finditer(r"[.!?]\s+", tail[half:]))
        if matches:
            cut = half + matches[0].end()
            overlap = tail[cut:].strip()
        else:
            overlap = tail[-self.overlap_chars:].strip()

        # Si l'overlap commence à l'intérieur d'une référence d'image markdown,
        # on coupe après pour ne pas avoir un fragment cassé.
        broken_image = re.search(r"\]\(rag-image://[^)]+\)", overlap)
        if broken_image:
            opening = overlap.rfind("![", 0, broken_image.start())
            if opening == -1:
                # Pas d'ouverture : on enlève le fragment cassé
                overlap = overlap[broken_image.end():].lstrip()
        return overlap

    # ------------------------------------------------------------------
    # Fusion des chunks trop courts
    # ------------------------------------------------------------------

    def _merge_tiny_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Fusionne les chunks < min_chunk_chars avec le chunk suivant."""
        if len(chunks) <= 1:
            return chunks

        result: List[Chunk] = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            # Si court et pas le dernier, fusionner
            while (len(current.text) < self.min_chunk_chars
                   and i + 1 < len(chunks)):
                next_chunk = chunks[i + 1]
                # Ne fusionne que si le total reste sous target_chars
                if len(current.text) + len(next_chunk.text) > self.target_chars * 1.5:
                    break
                current = Chunk(
                    text=f"{current.text}\n\n{next_chunk.text}",
                    heading_path=current.heading_path,
                    order=current.order,
                    char_count=len(current.text) + len(next_chunk.text),
                )
                i += 1
            result.append(current)
            i += 1

        # Renumérotation
        for idx, c in enumerate(result):
            c.order = idx
        return result
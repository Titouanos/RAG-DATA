"""Chunking markdown-aware (porté du POC, `fichier:ligne` d'origine dans ETAT_DES_LIEUX).

Stratégie :
1. Découpage par hiérarchie de titres ATX (H1..H6) — chaque section devient un bloc.
2. Si un bloc dépasse la taille cible, resplit sur les paragraphes avec overlap.
3. Chaque chunk conserve le **chemin des titres parents** en préfixe :
   `[Doc Title > Section > Sous-section]\n\n<corps>` — le contexte voyage avec le chunk.
4. Les chunks trop courts sont fusionnés avec le suivant.

Le format de préfixe est conservé à l'identique (visible par les embeddings et le LLM).
Limitations connues (backlog, cf. ETAT_DES_LIEUX R6) : les blocs de code et les tableaux
markdown ne sont pas traités spécifiquement.
"""

from __future__ import annotations

import re

from rag_builder.core.models import Chunk

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownChunker:
    """Chunker qui respecte la structure markdown."""

    def __init__(
        self,
        target_tokens: int = 500,
        overlap_tokens: int = 60,
        min_chunk_chars: int = 100,
    ):
        # Approximation char <-> token : ~4 chars/token en français.
        self.target_chars = target_tokens * 4
        self.overlap_chars = overlap_tokens * 4
        self.min_chunk_chars = min_chunk_chars

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def chunk(self, markdown: str, doc_title: str | None = None) -> list[Chunk]:
        """Chunke un document markdown en liste de `Chunk`."""
        if not markdown or not markdown.strip():
            return []

        sections = self._split_by_headings(markdown)

        chunks: list[Chunk] = []
        order = 0
        for heading_path, body in sections:
            body = body.strip()
            if not body:
                continue

            context_path: list[str] = []
            if doc_title:
                context_path.append(doc_title)
            context_path.extend(heading_path)

            for piece in self._split_by_size(body):
                piece = piece.strip()
                if not piece:
                    continue
                header_prefix = " > ".join(context_path)
                text = f"[{header_prefix}]\n\n{piece}" if header_prefix else piece
                chunks.append(
                    Chunk(
                        text=text,
                        order=order,
                        heading_path=list(heading_path),
                        page_or_section=" > ".join(heading_path),
                        char_count=len(text),
                    )
                )
                order += 1

        return self._merge_tiny_chunks(chunks)

    # ------------------------------------------------------------------
    # Découpage par titres
    # ------------------------------------------------------------------

    def _split_by_headings(self, md: str) -> list[tuple[list[str], str]]:
        """Retourne [(heading_path, body), ...] en reconstruisant la hiérarchie."""
        lines = md.split("\n")
        sections: list[tuple[list[str], str]] = []
        stack: list[tuple[int, str]] = []  # [(level, title), ...]
        buffer: list[str] = []
        current_path: list[str] = []

        def flush() -> None:
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
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current_path = [t for _, t in stack]
            else:
                buffer.append(line)
        flush()

        if not sections:
            body = md.strip()
            if body:
                sections.append(([], body))
        return sections

    # ------------------------------------------------------------------
    # Redécoupage par taille
    # ------------------------------------------------------------------

    def _split_by_size(self, text: str) -> list[str]:
        """Si le texte dépasse target_chars, le découpe sur les paragraphes."""
        if len(text) <= self.target_chars:
            return [text]

        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) > self.target_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._hard_split(para))
                continue
            candidate = f"{current}\n\n{para}" if current else para
            if len(candidate) <= self.target_chars:
                current = candidate
            else:
                chunks.append(current.strip())
                overlap = self._extract_overlap(current)
                current = f"{overlap}\n\n{para}" if overlap else para

        if current.strip():
            chunks.append(current.strip())

        return self._repair_split_urls(chunks)

    @staticmethod
    def _repair_split_urls(chunks: list[str]) -> list[str]:
        """Évite de couper une URL http(s) en deux entre chunks consécutifs."""
        if len(chunks) < 2:
            return chunks

        url_re = re.compile(r"https?://[^\s\)\]\"'<>]+$")
        url_continuation_chars = set("/:?&=#%+~,;_-")

        repaired: list[str] = []
        i = 0
        chunks = list(chunks)
        while i < len(chunks):
            current = chunks[i]
            if i + 1 < len(chunks):
                m = url_re.search(current)
                next_chunk = chunks[i + 1]
                if m and next_chunk and next_chunk[0] in url_continuation_chars:
                    url_start = m.start()
                    truncated_url = current[url_start:]
                    current = current[:url_start].rstrip()
                    chunks[i + 1] = f"{truncated_url}{next_chunk}"
            if current.strip():
                repaired.append(current)
            i += 1
        return repaired

    def _hard_split(self, text: str) -> list[str]:
        """Split brutal d'un texte > target_chars (essaie d'abord les phrases)."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            candidate = f"{current} {sent}" if current else sent
            if len(candidate) <= self.target_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(sent) > self.target_chars:
                    step = self.target_chars - self.overlap_chars
                    for i in range(0, len(sent), step):
                        chunks.append(sent[i : i + self.target_chars])
                    current = ""
                else:
                    current = sent
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _extract_overlap(self, text: str) -> str:
        """Prend ~overlap_chars de la fin du texte, aligné sur une frontière de phrase.

        Protège les références d'images markdown `![...](rag-image://...)` d'une coupure.
        """
        if self.overlap_chars <= 0 or not text:
            return ""
        tail = text[-self.overlap_chars * 2 :]
        half = len(tail) // 2
        matches = list(re.finditer(r"[.!?]\s+", tail[half:]))
        if matches:
            cut = half + matches[0].end()
            overlap = tail[cut:].strip()
        else:
            overlap = tail[-self.overlap_chars :].strip()

        broken_image = re.search(r"\]\(rag-image://[^)]+\)", overlap)
        if broken_image:
            opening = overlap.rfind("![", 0, broken_image.start())
            if opening == -1:
                overlap = overlap[broken_image.end() :].lstrip()
        return overlap

    # ------------------------------------------------------------------
    # Fusion des chunks trop courts
    # ------------------------------------------------------------------

    def _merge_tiny_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fusionne un chunk < min_chunk_chars avec le suivant, plafonné à 1.5×target."""
        if len(chunks) <= 1:
            return chunks

        result: list[Chunk] = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            while len(current.text) < self.min_chunk_chars and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                if len(current.text) + len(next_chunk.text) > self.target_chars * 1.5:
                    break
                merged_text = f"{current.text}\n\n{next_chunk.text}"
                current = Chunk(
                    text=merged_text,
                    order=current.order,
                    heading_path=current.heading_path,
                    page_or_section=current.page_or_section,
                    char_count=len(merged_text),
                )
                i += 1
            result.append(current)
            i += 1

        for idx, c in enumerate(result):
            c.order = idx
        return result

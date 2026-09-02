"""Chunking. The part of a RAG pipeline that decides its ceiling.

Retrieval cannot return a good passage that chunking never produced. Splitting
every 500 characters is the default and it is wrong in a specific way: it cuts
mid-sentence, so a chunk begins with half a clause and ends with half another,
and the embedding is computed over that fragment. The vector is a blend of two
incomplete thoughts and matches neither.

Three decisions here, each with a cost:

  Split on structure first, size second. Paragraphs and headings are authored
  boundaries - somebody already decided those ideas belong together.

  Overlap. A sentence near a boundary belongs to both neighbours; without
  overlap the answer sitting across a seam is retrievable from neither side.
  The cost is storage and some duplicate hits, which reranking then collapses.

  Carry the heading into the chunk. "Within 30 days" is useless without
  "Refunds". Retrieval sees only the chunk text, so anything a human would
  have scrolled up to read has to be inside it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CHARS = 700
OVERLAP_CHARS = 120
MIN_CHARS = 120


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    heading: str = ""
    index: int = 0

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded.

        The heading is prepended so a chunk saying "within 30 days" still
        carries "Refunds" into its vector. Without this, that chunk is
        unreachable by anyone who asks about refunds - which is everyone.
        """
        return f"{self.heading}\n\n{self.text}".strip() if self.heading else self.text


def chunk_document(text: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        section_start = len(chunks)
        for piece in _split_body(body):
            piece = piece.strip()
            if not piece:
                continue

            # Merge a too-small fragment backwards, but only into a chunk from
            # the SAME section. An earlier version merged across the section
            # boundary, so a short "Returns" section was absorbed into the
            # previous "Refunds" chunk and inherited its heading - the merged
            # text then embedded under the wrong topic and was unreachable by
            # anyone asking about returns. A short section is still a section.
            too_small = len(piece) < MIN_CHARS
            can_merge = len(chunks) > section_start

            if too_small and can_merge:
                previous = chunks[-1]
                chunks[-1] = Chunk(text=f"{previous.text} {piece}",
                                   source=source, heading=previous.heading,
                                   index=previous.index)
            else:
                chunks.append(Chunk(text=piece, source=source,
                                    heading=heading, index=len(chunks)))
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown headings, keeping each heading with its body."""
    parts = re.split(r"^(#{1,6}\s+.*)$", text, flags=re.M)
    if len(parts) == 1:
        return [("", text)]
    sections, heading = [], ""
    if parts[0].strip():
        sections.append(("", parts[0]))
    for i in range(1, len(parts), 2):
        heading = re.sub(r"^#+\s*", "", parts[i]).strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((heading, body))
    return sections


def _split_body(body: str) -> list[str]:
    """Pack paragraphs up to the target, then overlap by whole sentences.

    Overlapping by characters would reintroduce exactly the mid-sentence cut
    this is trying to avoid, so the overlap is taken as trailing sentences.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out, current = [], ""

    for para in paragraphs:
        if len(para) > TARGET_CHARS:
            if current:
                out.append(current)
                current = ""
            out.extend(_split_long_paragraph(para))
            continue
        if len(current) + len(para) + 2 <= TARGET_CHARS:
            current = f"{current}\n\n{para}" if current else para
        else:
            out.append(current)
            current = _tail_sentences(current, OVERLAP_CHARS) + "\n\n" + para

    if current:
        out.append(current)
    return out


def _split_long_paragraph(para: str) -> list[str]:
    sentences = _sentences(para)
    out, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= TARGET_CHARS:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                out.append(current)
            current = (_tail_sentences(current, OVERLAP_CHARS) + " " + sentence).strip()
    if current:
        out.append(current)
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _tail_sentences(text: str, budget: int) -> str:
    """Trailing whole sentences within the overlap budget."""
    kept: list[str] = []
    for sentence in reversed(_sentences(text)):
        if sum(len(s) for s in kept) + len(sentence) > budget:
            break
        kept.insert(0, sentence)
    return " ".join(kept)

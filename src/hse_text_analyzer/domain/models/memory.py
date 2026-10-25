"""
Agent memory domain models.

DocumentMemory is the agent's accumulated knowledge about entities
seen in previously processed sentences. It is built incrementally
as the linguistic analysis walks through the document, and is
attached to the Document so that any detector can read it.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EntityMention(BaseModel):
    """A single noun/proper-noun token extracted from one sentence."""
    lemma: str
    original_text: str
    gender: str | None = None
    number: str | None = None
    pos: str  # "NOUN" or "PROPN"
    sentence_idx: int


class DocumentMemory(BaseModel):
    """
    Accumulated cross-sentence entity knowledge for a document.

    Populated sentence-by-sentence by MemoryBuilder.  Detectors can
    call get_context_before_sentence() to retrieve a human-readable
    summary of every entity introduced before sentence N.
    """
    entity_mentions: list[EntityMention] = Field(default_factory=list)

    def get_context_before_sentence(self, sentence_idx: int) -> str:
        """
        Return a formatted string listing all unique entities (by lemma)
        that appeared in sentences strictly before *sentence_idx*.

        Returns an empty string if no prior entities exist.
        """
        prior = [e for e in self.entity_mentions if e.sentence_idx < sentence_idx]
        if not prior:
            return ""

        # Deduplicate by lemma, keep latest metadata
        seen: dict[str, EntityMention] = {}
        for e in prior:
            seen[e.lemma] = e

        lines = []
        for lemma, e in seen.items():
            parts = [f"«{e.original_text}» (лемма: {lemma}, POS: {e.pos}"]
            if e.gender:
                parts.append(f", род: {e.gender}")
            if e.number:
                parts.append(f", число: {e.number}")
            parts.append(f", предложение #{e.sentence_idx + 1})")
            lines.append("".join(parts))

        header = (
            f"Сущности, упомянутые до текущего предложения "
            f"(предложение #{sentence_idx + 1}):\n"
        )
        return header + "\n".join(f"  - {line}" for line in lines)

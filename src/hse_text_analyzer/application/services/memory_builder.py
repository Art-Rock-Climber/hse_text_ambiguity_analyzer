"""
MemoryBuilder — constructs a DocumentMemory from a completed LinguisticAnalysis.

Called by AnalyzeDocumentUseCase immediately after the linguistic analysis
stage, before the detector registry runs.  The memory is attached to the
Document so every detector can read it without changing the detect() signature.
"""
from __future__ import annotations

from hse_text_analyzer.domain.models.document import Document
from hse_text_analyzer.domain.models.memory import DocumentMemory, EntityMention

_ENTITY_POS = {"NOUN", "PROPN"}


class MemoryBuilder:
    """Builds a DocumentMemory from a Document's LinguisticAnalysis."""

    def build(self, document: Document) -> DocumentMemory:
        """
        Walk every sentence in the linguistic analysis and collect all
        NOUN / PROPN tokens into a flat list of EntityMention records,
        preserving sentence order.
        """
        memory = DocumentMemory()

        if not document.linguistic_analysis:
            return memory

        for sent_idx, sentence in enumerate(document.linguistic_analysis.sentences):
            for token in sentence.tokens:
                m = token.morphology
                if not m or m.pos not in _ENTITY_POS:
                    continue
                if not m.lemma:
                    continue

                memory.entity_mentions.append(
                    EntityMention(
                        lemma=m.lemma.lower(),
                        original_text=token.text,
                        gender=m.gender,
                        number=m.number,
                        pos=m.pos,
                        sentence_idx=sent_idx,
                    )
                )

        return memory

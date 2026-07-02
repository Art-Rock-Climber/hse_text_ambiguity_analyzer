from __future__ import annotations

import re
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, Doc

from hse_text_analyzer.application.ports.linguistic_analyzer import LinguisticAnalyzer
from hse_text_analyzer.domain.models.document import Document
from hse_text_analyzer.domain.models.linguistic import LinguisticAnalysis, MorphologicalFeatures, Sentence, Token
from hse_text_analyzer.infrastructure.nlp.pymorphy_adapter import PymorphyMorphologyAdapter


class RussianLinguisticAnalyzer(LinguisticAnalyzer):
    name = "russian_linguistic_analyzer"

    def __init__(self) -> None:
        self._morphology_adapter = PymorphyMorphologyAdapter()
        self._segmenter = Segmenter()
        self._emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._emb)
        self._syntax_parser = NewsSyntaxParser(self._emb)

    def supports(self, document: Document) -> bool:
        return bool(document.raw_text and document.raw_text.strip())

    def analyze(self, document: Document) -> Document:
        if not document.raw_text or not document.raw_text.strip():
            analysis = LinguisticAnalysis(
                document_id=document.id,
                sentence_count=0,
                token_count=0,
                sentences=[],
                summary={"sentences": 0, "tokens": 0},
            )
            return document.model_copy(update={"linguistic_analysis": analysis})

        doc = Doc(document.raw_text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)
        doc.parse_syntax(self._syntax_parser)

        sentences = []
        for sent in doc.sents:
            tokens: list[Token] = []
            
            # Map Natasha token id to its index in this sentence
            id_to_idx = {t.id: idx for idx, t in enumerate(sent.tokens)}
            
            for t in sent.tokens:
                # Determine punctuation
                is_punct = t.pos == "PUNCT" or bool(re.fullmatch(r"[^\w\s]", t.text, re.UNICODE))
                morphology = (
                    self._morphology_adapter.analyze(t.text)
                    if not is_punct
                    else MorphologicalFeatures()
                )
                
                tokens.append(
                    Token(
                        text=t.text,
                        start_char=t.start,
                        end_char=t.stop,
                        is_punctuation=is_punct,
                        morphology=morphology,
                    )
                )
                
            # Connect syntactic parent and relation
            for idx, t in enumerate(sent.tokens):
                parent_idx = None
                relation = None
                if t.head_id and t.head_id in id_to_idx:
                    parent_idx = id_to_idx[t.head_id]
                if t.rel:
                    relation = t.rel
                
                tokens[idx].syntax_parent_idx = parent_idx
                tokens[idx].syntax_relation = relation
                
            sentences.append(
                Sentence(
                    text=sent.text,
                    start_char=sent.start,
                    end_char=sent.stop,
                    tokens=tokens,
                )
            )

        analysis = LinguisticAnalysis(
            document_id=document.id,
            sentence_count=len(sentences),
            token_count=sum(len(s.tokens) for s in sentences),
            sentences=sentences,
            summary={
                "sentences": len(sentences),
                "tokens": sum(len(s.tokens) for s in sentences),
            },
        )
        return document.model_copy(update={"linguistic_analysis": analysis})

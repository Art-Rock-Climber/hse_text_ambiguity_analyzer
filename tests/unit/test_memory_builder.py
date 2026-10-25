import pytest
from hse_text_analyzer.application.services.memory_builder import MemoryBuilder
from hse_text_analyzer.domain.models.document import Document, DocumentMetadata, DocumentSection
from hse_text_analyzer.domain.enums import SourceFileType, DocumentType
from hse_text_analyzer.domain.models.linguistic import (
    LinguisticAnalysis, MorphologicalFeatures, Sentence, Token
)


def _make_document(sentences: list[Sentence]) -> Document:
    analysis = LinguisticAnalysis(
        document_id="test-doc",
        sentence_count=len(sentences),
        token_count=sum(len(s.tokens) for s in sentences),
        sentences=sentences,
        summary={},
    )
    return Document(
        title="test",
        raw_text="тест",
        metadata=DocumentMetadata(
            file_name="test.txt",
            file_type=SourceFileType.TXT,
            document_type=DocumentType.OTHER,
            parser_name="test",
        ),
        linguistic_analysis=analysis,
    )


def _noun_token(text: str, lemma: str, gender: str = "Masc", number: str = "Sing") -> Token:
    return Token(
        text=text,
        start_char=0,
        end_char=len(text),
        is_punctuation=False,
        morphology=MorphologicalFeatures(pos="NOUN", lemma=lemma, gender=gender, number=number),
    )


def _verb_token(text: str) -> Token:
    return Token(
        text=text,
        start_char=0,
        end_char=len(text),
        is_punctuation=False,
        morphology=MorphologicalFeatures(pos="VERB"),
    )


def test_memory_builder_extracts_nouns() -> None:
    sentences = [
        Sentence(text="Студент пишет.", start_char=0, end_char=14, tokens=[
            _noun_token("Студент", "студент"),
            _verb_token("пишет"),
        ]),
    ]
    doc = _make_document(sentences)
    memory = MemoryBuilder().build(doc)

    assert len(memory.entity_mentions) == 1
    assert memory.entity_mentions[0].lemma == "студент"
    assert memory.entity_mentions[0].sentence_idx == 0


def test_memory_builder_ignores_verbs_and_punctuation() -> None:
    verb = _verb_token("читает")
    punct = Token(text=".", start_char=7, end_char=8, is_punctuation=True,
                  morphology=MorphologicalFeatures())
    noun = _noun_token("Профессор", "профессор")
    sentences = [Sentence(text="Профессор читает.", start_char=0, end_char=17,
                          tokens=[noun, verb, punct])]
    doc = _make_document(sentences)
    memory = MemoryBuilder().build(doc)

    assert all(e.pos in {"NOUN", "PROPN"} for e in memory.entity_mentions)
    assert len(memory.entity_mentions) == 1


def test_context_before_sentence_returns_empty_for_first_sentence() -> None:
    sentences = [
        Sentence(text="Мария пришла.", start_char=0, end_char=13, tokens=[
            _noun_token("Мария", "мария", gender="Fem"),
        ]),
    ]
    doc = _make_document(sentences)
    memory = MemoryBuilder().build(doc)

    ctx = memory.get_context_before_sentence(0)
    assert ctx == ""


def test_context_before_sentence_returns_prior_entities() -> None:
    sentences = [
        Sentence(text="Мария и Анна работали.", start_char=0, end_char=22, tokens=[
            _noun_token("Мария", "мария", gender="Fem"),
            _noun_token("Анна", "анна", gender="Fem"),
        ]),
        Sentence(text="Она опоздала.", start_char=23, end_char=36, tokens=[
            _noun_token("Она", "она"),
        ]),
    ]
    doc = _make_document(sentences)
    memory = MemoryBuilder().build(doc)

    ctx = memory.get_context_before_sentence(1)
    assert "мария" in ctx.lower() or "Мария" in ctx
    assert "анна" in ctx.lower() or "Анна" in ctx


def test_memory_builder_returns_empty_memory_when_no_linguistic_analysis() -> None:
    doc = Document(
        title="test",
        raw_text="",
        metadata=DocumentMetadata(
            file_name="test.txt",
            file_type=SourceFileType.TXT,
            document_type=DocumentType.OTHER,
            parser_name="test",
        ),
    )
    memory = MemoryBuilder().build(doc)
    assert memory.entity_mentions == []

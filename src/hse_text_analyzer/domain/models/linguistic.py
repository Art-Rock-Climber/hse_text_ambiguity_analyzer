from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class MorphologicalFeatures(BaseModel):
    lemma: str | None = None
    pos: str | None = None
    tag: str | None = None
    grammemes: list[str] = Field(default_factory=list)
    normal_form: str | None = None
    is_known: bool = False
    gender: str | None = None
    number: str | None = None
    case: str | None = None
    tense: str | None = None
    aspect: str | None = None
    person: str | None = None
    mood: str | None = None
    voice: str | None = None


class Token(BaseModel):
    text: str
    start_char: int
    end_char: int
    is_punctuation: bool = False
    morphology: MorphologicalFeatures = Field(default_factory=MorphologicalFeatures)
    syntax_parent_idx: int | None = None
    syntax_relation: str | None = None


class Sentence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    start_char: int
    end_char: int
    tokens: list[Token] = Field(default_factory=list)


class LinguisticAnalysis(BaseModel):
    document_id: str
    language: str = "ru"
    sentence_count: int = 0
    token_count: int = 0
    sentences: list[Sentence] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary: dict[str, int] = Field(default_factory=dict)

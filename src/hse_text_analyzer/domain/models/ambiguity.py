from uuid import uuid4

from pydantic import BaseModel, Field

from hse_text_analyzer.domain.enums import AmbiguityCategory, FindingSource, Severity


class TextSpan(BaseModel):
    document_id: str
    start_char: int
    end_char: int
    text: str
    section_id: str | None = None


class Evidence(BaseModel):
    message: str
    span: TextSpan | None = None
    features: dict[str, str | int | float | bool] = Field(default_factory=dict)


class AmbiguityFinding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    span: TextSpan
    category: AmbiguityCategory
    source: FindingSource
    detector_id: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    title: str
    description: str
    evidence: list[Evidence] = Field(default_factory=list)
    suggested_revision: str | None = None
    explanation: str | None = None

from pydantic import BaseModel, Field


class AISyntaxFinding(BaseModel):
    title: str
    description: str
    ambiguous_fragment: str
    alternative_attachment: str
    explanation: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_revision: str | None = None
    
    # Programmatic structural override details
    ambiguous_token_idx: int | None = None
    alternative_parent_idx: int | None = None
    alternative_relation: str | None = None


class AISyntaxResponse(BaseModel):
    is_ambiguous: bool
    findings: list[AISyntaxFinding] = Field(default_factory=list)
    error_message: str | None = None


class AIReferentialFinding(BaseModel):
    title: str
    description: str
    ambiguous_fragment: str
    possible_referents: list[str] = Field(default_factory=list)
    explanation: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_revision: str | None = None


class AIReferentialResponse(BaseModel):
    is_ambiguous: bool
    findings: list[AIReferentialFinding] = Field(default_factory=list)
    error_message: str | None = None


class AICoordinationalFinding(BaseModel):
    title: str
    description: str
    ambiguous_fragment: str
    explanation: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_revision: str | None = None


class AICoordinationalResponse(BaseModel):
    is_ambiguous: bool
    findings: list[AICoordinationalFinding] = Field(default_factory=list)
    error_message: str | None = None

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from hse_text_analyzer.domain.models.ambiguity import AmbiguityFinding


class AnalysisSummary(BaseModel):
    total_findings: int = 0
    rule_based_findings: int = 0
    ai_findings: int = 0
    parser_warnings: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    document_id: str
    findings: list[AmbiguityFinding] = Field(default_factory=list)
    summary: AnalysisSummary = Field(default_factory=AnalysisSummary)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

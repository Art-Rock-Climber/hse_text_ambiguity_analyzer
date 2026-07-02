from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from hse_text_analyzer.domain.enums import DocumentType, SourceFileType
from hse_text_analyzer.domain.models.linguistic import LinguisticAnalysis


class DocumentMetadata(BaseModel):
    file_name: str
    file_type: SourceFileType
    document_type: DocumentType = DocumentType.OTHER
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parser_name: str
    warnings: list[str] = Field(default_factory=list)


class DocumentSection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str | None = None
    level: int = 1
    text: str
    start_char: int = 0
    end_char: int = 0


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    raw_text: str
    metadata: DocumentMetadata
    sections: list[DocumentSection] = Field(default_factory=list)
    linguistic_analysis: LinguisticAnalysis | None = None

    @property
    def character_count(self) -> int:
        return len(self.raw_text)

    @property
    def word_count(self) -> int:
        return len(self.raw_text.split())

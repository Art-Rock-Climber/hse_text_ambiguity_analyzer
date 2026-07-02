from abc import ABC, abstractmethod
from dataclasses import dataclass

from hse_text_analyzer.domain.enums import AmbiguityCategory, DocumentType
from hse_text_analyzer.domain.models.ambiguity import AmbiguityFinding
from hse_text_analyzer.domain.models.document import Document


@dataclass(frozen=True)
class DetectorMetadata:
    detector_id: str
    name: str
    description: str
    version: str
    categories: tuple[AmbiguityCategory, ...]
    supported_document_types: tuple[DocumentType, ...] = (DocumentType.OTHER,)
    default_enabled: bool = True


class AmbiguityDetector(ABC):
    metadata: DetectorMetadata

    def supports(self, document: Document) -> bool:
        supported_types = self.metadata.supported_document_types
        return (
            DocumentType.OTHER in supported_types
            or document.metadata.document_type in supported_types
        )

    @abstractmethod
    def detect(self, document: Document) -> list[AmbiguityFinding]:
        """Return ambiguity findings for a document."""

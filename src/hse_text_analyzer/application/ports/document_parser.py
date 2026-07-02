from abc import ABC, abstractmethod
from dataclasses import dataclass

from hse_text_analyzer.domain.enums import DocumentType, SourceFileType
from hse_text_analyzer.domain.models.document import Document


@dataclass(frozen=True)
class UploadedDocument:
    file_name: str
    content: bytes
    declared_document_type: DocumentType = DocumentType.OTHER

    @property
    def extension(self) -> str:
        return self.file_name.rsplit(".", maxsplit=1)[-1].lower() if "." in self.file_name else ""


class DocumentParser(ABC):
    name: str
    supported_file_types: frozenset[SourceFileType]

    def supports(self, uploaded_document: UploadedDocument) -> bool:
        return uploaded_document.extension in self.supported_file_types

    @abstractmethod
    def parse(self, uploaded_document: UploadedDocument) -> Document:
        """Parse an uploaded file into a normalized domain document."""

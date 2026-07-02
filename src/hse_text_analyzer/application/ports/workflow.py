from abc import ABC, abstractmethod

from hse_text_analyzer.application.ports.document_parser import UploadedDocument
from hse_text_analyzer.domain.models.analysis import AnalysisResult
from hse_text_analyzer.domain.models.document import Document


class AnalysisWorkflow(ABC):
    @abstractmethod
    def run(self, uploaded_document: UploadedDocument) -> tuple[Document, AnalysisResult]:
        """Run the document analysis workflow."""

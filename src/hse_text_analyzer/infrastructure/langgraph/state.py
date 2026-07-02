from typing import TypedDict

from hse_text_analyzer.application.ports.document_parser import UploadedDocument
from hse_text_analyzer.domain.models.analysis import AnalysisResult
from hse_text_analyzer.domain.models.document import Document


class AnalysisState(TypedDict, total=False):
    uploaded_document: UploadedDocument
    document: Document
    analysis_result: AnalysisResult

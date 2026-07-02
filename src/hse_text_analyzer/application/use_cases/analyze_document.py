from hse_text_analyzer.application.ports.document_parser import UploadedDocument
from hse_text_analyzer.application.ports.linguistic_analyzer import LinguisticAnalyzer
from hse_text_analyzer.application.services.detector_registry import DetectorRegistry
from hse_text_analyzer.application.services.document_loading import DocumentLoadingService
from hse_text_analyzer.application.services.result_aggregator import ResultAggregator
from hse_text_analyzer.domain.models.analysis import AnalysisResult
from hse_text_analyzer.domain.models.document import Document


class AnalyzeDocumentUseCase:
    def __init__(
        self,
        document_loader: DocumentLoadingService,
        detector_registry: DetectorRegistry,
        result_aggregator: ResultAggregator,
        linguistic_analyzer: LinguisticAnalyzer,
    ) -> None:
        self._document_loader = document_loader
        self._detector_registry = detector_registry
        self._result_aggregator = result_aggregator
        self._linguistic_analyzer = linguistic_analyzer

    def execute(self, uploaded_document: UploadedDocument) -> tuple[Document, AnalysisResult]:
        document = self._document_loader.load(uploaded_document)
        document = self._linguistic_analyzer.analyze(document)
        findings = self._detector_registry.detect(document)
        result = self._result_aggregator.aggregate(document, findings)
        return document, result

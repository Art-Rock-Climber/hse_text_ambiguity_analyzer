from hse_text_analyzer.application.services.detector_registry import DetectorRegistry
from hse_text_analyzer.application.services.document_loading import DocumentLoadingService
from hse_text_analyzer.application.services.result_aggregator import ResultAggregator
from hse_text_analyzer.application.use_cases.analyze_document import AnalyzeDocumentUseCase
from hse_text_analyzer.infrastructure.langgraph import LangGraphAnalysisWorkflow
from hse_text_analyzer.infrastructure.nlp import RussianLinguisticAnalyzer
from hse_text_analyzer.infrastructure.parsing import (
    DocxDocumentParser,
    MarkdownDocumentParser,
    PdfDocumentParser,
    TextDocumentParser,
)
from hse_text_analyzer.plugins.detectors import built_in_detectors


def build_analysis_workflow() -> LangGraphAnalysisWorkflow:
    parsers = [
        TextDocumentParser(),
        MarkdownDocumentParser(),
        DocxDocumentParser(),
        PdfDocumentParser(),
    ]
    detector_registry = DetectorRegistry()
    detector_registry.register_many(built_in_detectors())
    use_case = AnalyzeDocumentUseCase(
        document_loader=DocumentLoadingService(parsers),
        detector_registry=detector_registry,
        result_aggregator=ResultAggregator(),
        linguistic_analyzer=RussianLinguisticAnalyzer(),
    )
    return LangGraphAnalysisWorkflow(use_case)

from hse_text_analyzer.application.ports.document_parser import UploadedDocument
from hse_text_analyzer.application.ports.linguistic_analyzer import LinguisticAnalyzer
from hse_text_analyzer.application.services.detector_registry import DetectorRegistry
from hse_text_analyzer.application.services.document_loading import DocumentLoadingService
from hse_text_analyzer.application.services.memory_builder import MemoryBuilder
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
        memory_builder: MemoryBuilder | None = None,
    ) -> None:
        self._document_loader = document_loader
        self._detector_registry = detector_registry
        self._result_aggregator = result_aggregator
        self._linguistic_analyzer = linguistic_analyzer
        self._memory_builder = memory_builder or MemoryBuilder()

    def execute(self, uploaded_document: UploadedDocument) -> tuple[Document, AnalysisResult]:
        # Stage 1: parse raw file → domain Document
        document = self._document_loader.load(uploaded_document)

        # Stage 2: NLP — segmentation, morphology, syntax parse
        document = self._linguistic_analyzer.analyze(document)

        # Stage 3: build agent memory from the completed linguistic analysis
        #           (cross-sentence entity knowledge for the referential detector)
        agent_memory = self._memory_builder.build(document)
        document = document.model_copy(update={"agent_memory": agent_memory})

        # Stage 4: run ambiguity detectors (prefilter → AI agents)
        findings = self._detector_registry.detect(document)

        # Stage 5: aggregate findings into AnalysisResult
        result = self._result_aggregator.aggregate(document, findings)
        return document, result

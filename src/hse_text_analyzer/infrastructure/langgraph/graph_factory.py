from hse_text_analyzer.application.ports.document_parser import UploadedDocument
from hse_text_analyzer.application.ports.workflow import AnalysisWorkflow
from hse_text_analyzer.application.use_cases.analyze_document import AnalyzeDocumentUseCase
from hse_text_analyzer.domain.models.analysis import AnalysisResult
from hse_text_analyzer.domain.models.document import Document
from hse_text_analyzer.infrastructure.langgraph.state import AnalysisState


class LangGraphAnalysisWorkflow(AnalysisWorkflow):
    def __init__(self, analyze_document_use_case: AnalyzeDocumentUseCase) -> None:
        self._analyze_document_use_case = analyze_document_use_case
        self._graph = self._build_graph()

    def run(self, uploaded_document: UploadedDocument) -> tuple[Document, AnalysisResult]:
        final_state = self._graph.invoke({"uploaded_document": uploaded_document})
        return final_state["document"], final_state["analysis_result"]

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("langgraph is required to run the analysis workflow.") from exc

        graph = StateGraph(AnalysisState)
        graph.add_node("load_and_analyze_document", self._load_and_analyze_document)
        graph.add_edge(START, "load_and_analyze_document")
        graph.add_edge("load_and_analyze_document", END)
        return graph.compile()

    def _load_and_analyze_document(self, state: AnalysisState) -> AnalysisState:
        document, analysis_result = self._analyze_document_use_case.execute(
            state["uploaded_document"]
        )
        return {
            "document": document,
            "analysis_result": analysis_result,
        }

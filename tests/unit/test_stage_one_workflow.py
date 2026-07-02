from hse_text_analyzer.application.ports.document_parser import UploadedDocument
from hse_text_analyzer.application.services.composition import build_analysis_workflow


def test_stage_one_workflow_loads_document_and_returns_empty_analysis() -> None:
    workflow = build_analysis_workflow()
    uploaded = UploadedDocument(
        file_name="essay.txt",
        content="Это первый тестовый документ.".encode(),
    )

    document, result = workflow.run(uploaded)

    assert document.raw_text == "Это первый тестовый документ."
    assert result.document_id == document.id
    assert result.summary.total_findings == 0
    assert document.linguistic_analysis is not None
    assert document.linguistic_analysis.sentence_count == 1
    assert document.linguistic_analysis.token_count >= 4

from hse_text_analyzer.application.ports.document_parser import UploadedDocument
from hse_text_analyzer.domain.enums import DocumentType
from hse_text_analyzer.infrastructure.parsing import TextDocumentParser


def test_text_parser_builds_document_model() -> None:
    parser = TextDocumentParser()
    uploaded = UploadedDocument(
        file_name="assignment.txt",
        content="Проведите анализ текста.".encode(),
        declared_document_type=DocumentType.ASSIGNMENT,
    )

    document = parser.parse(uploaded)

    assert document.title == "assignment.txt"
    assert document.raw_text == "Проведите анализ текста."
    assert document.metadata.document_type == DocumentType.ASSIGNMENT
    assert document.metadata.parser_name == "plain_text_parser"
    assert document.sections[0].end_char == len(document.raw_text)

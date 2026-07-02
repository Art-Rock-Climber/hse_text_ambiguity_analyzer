from hse_text_analyzer.application.ports.document_parser import DocumentParser, UploadedDocument
from hse_text_analyzer.domain.exceptions import UnsupportedDocumentTypeError
from hse_text_analyzer.domain.models.document import Document


class DocumentLoadingService:
    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._parsers = parsers

    def load(self, uploaded_document: UploadedDocument) -> Document:
        for parser in self._parsers:
            if parser.supports(uploaded_document):
                return parser.parse(uploaded_document)

        supported = sorted(
            file_type.value
            for parser in self._parsers
            for file_type in parser.supported_file_types
        )
        raise UnsupportedDocumentTypeError(
            f"Unsupported file type '.{uploaded_document.extension}'. "
            f"Supported types: {', '.join(supported)}."
        )

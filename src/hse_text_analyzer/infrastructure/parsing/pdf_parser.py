from hse_text_analyzer.application.ports.document_parser import DocumentParser, UploadedDocument
from hse_text_analyzer.domain.enums import SourceFileType
from hse_text_analyzer.domain.exceptions import DocumentParsingError
from hse_text_analyzer.domain.models.document import Document, DocumentMetadata, DocumentSection


class PdfDocumentParser(DocumentParser):
    name = "pdf_parser"
    supported_file_types = frozenset({SourceFileType.PDF})

    def parse(self, uploaded_document: UploadedDocument) -> Document:
        try:
            import fitz
        except ImportError as exc:
            raise DocumentParsingError("PyMuPDF is required to parse PDF files.") from exc

        try:
            pdf = fitz.open(stream=uploaded_document.content, filetype="pdf")
            page_texts = [page.get_text("text").strip() for page in pdf]
        except Exception as exc:
            message = f"Could not parse PDF file '{uploaded_document.file_name}'."
            raise DocumentParsingError(message) from exc

        text = "\n\n".join(page_text for page_text in page_texts if page_text)
        warnings = []
        if not text:
            warnings.append("No selectable text was found in the PDF.")

        return Document(
            title=uploaded_document.file_name,
            raw_text=text,
            metadata=DocumentMetadata(
                file_name=uploaded_document.file_name,
                file_type=SourceFileType.PDF,
                document_type=uploaded_document.declared_document_type,
                parser_name=self.name,
                warnings=warnings,
            ),
            sections=[
                DocumentSection(title=None, text=text, start_char=0, end_char=len(text))
            ] if text else [],
        )

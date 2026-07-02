from hse_text_analyzer.application.ports.document_parser import DocumentParser, UploadedDocument
from hse_text_analyzer.domain.enums import SourceFileType
from hse_text_analyzer.domain.models.document import Document, DocumentMetadata, DocumentSection


class BasePlainTextParser(DocumentParser):
    encoding_candidates = ("utf-8-sig", "utf-8", "cp1251")

    def _decode(self, content: bytes) -> tuple[str, list[str]]:
        warnings: list[str] = []
        for encoding in self.encoding_candidates:
            try:
                return content.decode(encoding), warnings
            except UnicodeDecodeError:
                continue

        warnings.append(
            "File encoding could not be detected reliably; replacement decoding was used."
        )
        return content.decode("utf-8", errors="replace"), warnings

    def parse(self, uploaded_document: UploadedDocument) -> Document:
        text, warnings = self._decode(uploaded_document.content)
        text = text.strip()
        return Document(
            title=uploaded_document.file_name,
            raw_text=text,
            metadata=DocumentMetadata(
                file_name=uploaded_document.file_name,
                file_type=SourceFileType(uploaded_document.extension),
                document_type=uploaded_document.declared_document_type,
                parser_name=self.name,
                warnings=warnings,
            ),
            sections=[
                DocumentSection(
                    title=None,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                )
            ]
            if text
            else [],
        )


class TextDocumentParser(BasePlainTextParser):
    name = "plain_text_parser"
    supported_file_types = frozenset({SourceFileType.TXT})


class MarkdownDocumentParser(BasePlainTextParser):
    name = "markdown_parser"
    supported_file_types = frozenset({SourceFileType.MARKDOWN})

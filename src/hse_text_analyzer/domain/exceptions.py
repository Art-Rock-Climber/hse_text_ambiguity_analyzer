class HSETextAnalyzerError(Exception):
    """Base exception for application-specific errors."""


class UnsupportedDocumentTypeError(HSETextAnalyzerError):
    """Raised when no parser can handle the provided document."""


class DocumentParsingError(HSETextAnalyzerError):
    """Raised when a supported document cannot be parsed."""


class PluginRegistrationError(HSETextAnalyzerError):
    """Raised when a plugin cannot be registered safely."""

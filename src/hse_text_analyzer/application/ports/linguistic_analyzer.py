from abc import ABC, abstractmethod

from hse_text_analyzer.domain.models.document import Document


class LinguisticAnalyzer(ABC):
    name: str

    @abstractmethod
    def supports(self, document: Document) -> bool:
        raise NotImplementedError

    @abstractmethod
    def analyze(self, document: Document) -> Document:
        raise NotImplementedError

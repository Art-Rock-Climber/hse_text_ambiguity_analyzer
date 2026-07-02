from collections.abc import Iterable

from hse_text_analyzer.application.ports.ambiguity_detector import AmbiguityDetector
from hse_text_analyzer.domain.exceptions import PluginRegistrationError
from hse_text_analyzer.domain.models.ambiguity import AmbiguityFinding
from hse_text_analyzer.domain.models.document import Document


class DetectorRegistry:
    def __init__(self) -> None:
        self._detectors: dict[str, AmbiguityDetector] = {}

    def register(self, detector: AmbiguityDetector) -> None:
        detector_id = detector.metadata.detector_id
        if detector_id in self._detectors:
            raise PluginRegistrationError(f"Detector '{detector_id}' is already registered.")
        self._detectors[detector_id] = detector

    def register_many(self, detectors: Iterable[AmbiguityDetector]) -> None:
        for detector in detectors:
            self.register(detector)

    def all(self) -> list[AmbiguityDetector]:
        return list(self._detectors.values())

    def enabled_for(self, document: Document) -> list[AmbiguityDetector]:
        return [
            detector
            for detector in self._detectors.values()
            if detector.metadata.default_enabled and detector.supports(document)
        ]

    def detect(self, document: Document) -> list[AmbiguityFinding]:
        findings: list[AmbiguityFinding] = []
        for detector in self.enabled_for(document):
            findings.extend(detector.detect(document))
        return findings

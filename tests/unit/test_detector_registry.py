import pytest

from hse_text_analyzer.application.ports.ambiguity_detector import (
    AmbiguityDetector,
    DetectorMetadata,
)
from hse_text_analyzer.application.services.detector_registry import DetectorRegistry
from hse_text_analyzer.domain.enums import AmbiguityCategory
from hse_text_analyzer.domain.exceptions import PluginRegistrationError
from hse_text_analyzer.domain.models.ambiguity import AmbiguityFinding
from hse_text_analyzer.domain.models.document import Document


class NoopDetector(AmbiguityDetector):
    metadata = DetectorMetadata(
        detector_id="noop",
        name="Noop",
        description="Detector used by registry tests.",
        version="0.1.0",
        categories=(AmbiguityCategory.LEXICAL_VAGUENESS,),
    )

    def detect(self, document: Document) -> list[AmbiguityFinding]:
        return []


def test_registry_rejects_duplicate_detector_ids() -> None:
    registry = DetectorRegistry()
    registry.register(NoopDetector())

    with pytest.raises(PluginRegistrationError):
        registry.register(NoopDetector())

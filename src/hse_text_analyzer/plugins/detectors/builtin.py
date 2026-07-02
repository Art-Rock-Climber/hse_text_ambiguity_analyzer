from hse_text_analyzer.application.ports.ambiguity_detector import AmbiguityDetector
from hse_text_analyzer.plugins.detectors.syntax_detector import SyntaxAmbiguityDetector
from hse_text_analyzer.plugins.detectors.referential_detector import ReferentialAmbiguityDetector
from hse_text_analyzer.plugins.detectors.coordinational_detector import CoordinationalAmbiguityDetector


def built_in_detectors() -> list[AmbiguityDetector]:
    return [
        SyntaxAmbiguityDetector(),
        ReferentialAmbiguityDetector(),
        CoordinationalAmbiguityDetector(),
    ]

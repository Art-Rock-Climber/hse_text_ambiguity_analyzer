from unittest.mock import patch
from hse_text_analyzer.domain.models.document import Document, DocumentMetadata
from hse_text_analyzer.domain.enums import SourceFileType, DocumentType, AmbiguityCategory, FindingSource
from hse_text_analyzer.domain.models.linguistic import LinguisticAnalysis, Sentence, Token
from hse_text_analyzer.plugins.detectors.coordinational_detector import CoordinationalAmbiguityDetector
from hse_text_analyzer.infrastructure.ai.schemas import AICoordinationalResponse, AICoordinationalFinding


def test_coordinational_ambiguity_detector_creates_findings_when_ambiguous() -> None:
    # 1. Setup mock response from OpenRouter
    mock_finding = AICoordinationalFinding(
        title="Координационная неоднозначность группировки элементов",
        description="Неясно, как группируются 'студенты и преподаватели' относительно 'и родители'",
        ambiguous_fragment="Студенты и преподаватели и родители",
        explanation="Пояснение...",
        confidence=0.85,
        suggested_revision="Студенты вместе с преподавателями, а также родители пришли на встречу."
    )
    mock_response = AICoordinationalResponse(
        is_ambiguous=True,
        findings=[mock_finding]
    )

    # 2. Setup document with linguistic analysis
    tokens = [
        Token(text="Студенты", start_char=0, end_char=8, is_punctuation=False),
        Token(text="и", start_char=9, end_char=10, is_punctuation=False),
        Token(text="преподаватели", start_char=11, end_char=24, is_punctuation=False),
        Token(text="и", start_char=25, end_char=26, is_punctuation=False),
        Token(text="родители", start_char=27, end_char=35, is_punctuation=False),
        Token(text="пришли", start_char=36, end_char=42, is_punctuation=False),
    ]
    
    sentence = Sentence(
        text="Студенты и преподаватели и родители пришли",
        start_char=0,
        end_char=42,
        tokens=tokens
    )
    
    analysis = LinguisticAnalysis(
        document_id="doc4",
        sentence_count=1,
        token_count=6,
        sentences=[sentence]
    )
    
    doc = Document(
        id="doc4",
        title="essay.txt",
        raw_text="Студенты и преподаватели и родители пришли",
        metadata=DocumentMetadata(
            file_name="essay.txt",
            file_type=SourceFileType.TXT,
            document_type=DocumentType.OTHER,
            parser_name="plain_text_parser",
        ),
        linguistic_analysis=analysis
    )

    detector = CoordinationalAmbiguityDetector()
    
    # Mock OpenRouterClient.analyze_sentence_coordination
    with patch.object(detector._client, "analyze_sentence_coordination", return_value=mock_response) as mock_analyze:
        findings = detector.detect(doc)
        
        # Verify OpenRouterClient was called with sentence text
        mock_analyze.assert_called_once()
        args, kwargs = mock_analyze.call_args
        assert args[0] == "Студенты и преподаватели и родители пришли"
        
        # Verify findings format
        assert len(findings) == 1
        f = findings[0]
        assert f.category == AmbiguityCategory.COORDINATIONAL_AMBIGUITY
        assert f.source == FindingSource.LLM_AGENT
        assert f.detector_id == "coordinational_ambiguity_detector"
        assert f.confidence == 0.85
        assert f.title == "Координационная неоднозначность группировки элементов"
        assert f.span.text == "Студенты и преподаватели и родители"
        assert f.span.start_char == 0
        assert f.span.end_char == 35
        
        # Verify evidence
        assert len(f.evidence) == 1
        ev = f.evidence[0]
        assert ev.message == "Ambiguous coordinate group: Студенты и преподаватели и родители"


def test_coordinational_ambiguity_detector_skips_when_not_complex() -> None:
    # Setup document with only 1 conjunction ("и") and no commas
    tokens = [
        Token(text="Студенты", start_char=0, end_char=8, is_punctuation=False),
        Token(text="и", start_char=9, end_char=10, is_punctuation=False),
        Token(text="преподаватели", start_char=11, end_char=24, is_punctuation=False),
        Token(text="пришли", start_char=25, end_char=31, is_punctuation=False),
    ]
    
    sentence = Sentence(
        text="Студенты и преподаватели пришли",
        start_char=0,
        end_char=31,
        tokens=tokens
    )
    
    analysis = LinguisticAnalysis(
        document_id="doc_pref2",
        sentence_count=1,
        token_count=4,
        sentences=[sentence]
    )
    
    doc = Document(
        id="doc_pref2",
        title="essay.txt",
        raw_text="Студенты и преподаватели пришли",
        metadata=DocumentMetadata(
            file_name="essay.txt",
            file_type=SourceFileType.TXT,
            document_type=DocumentType.OTHER,
            parser_name="plain_text_parser",
        ),
        linguistic_analysis=analysis
    )

    detector = CoordinationalAmbiguityDetector()
    
    with patch.object(detector._client, "analyze_sentence_coordination") as mock_analyze:
        findings = detector.detect(doc)
        
        assert len(findings) == 0
        mock_analyze.assert_not_called()

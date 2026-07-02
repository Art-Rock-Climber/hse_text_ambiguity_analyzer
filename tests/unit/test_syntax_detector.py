from unittest.mock import patch
from hse_text_analyzer.domain.models.document import Document, DocumentMetadata
from hse_text_analyzer.domain.enums import SourceFileType, DocumentType, AmbiguityCategory, FindingSource
from hse_text_analyzer.domain.models.linguistic import LinguisticAnalysis, Sentence, Token
from hse_text_analyzer.plugins.detectors.syntax_detector import SyntaxAmbiguityDetector
from hse_text_analyzer.infrastructure.ai.schemas import AISyntaxResponse, AISyntaxFinding


def test_syntax_ambiguity_detector_creates_findings_when_ambiguous() -> None:
    # 1. Setup mock response from OpenRouter
    mock_finding = AISyntaxFinding(
        title="Синтаксическая неоднозначность привязки предложной группы",
        description="Оборот 'с биноклем' может прикрепляться к глаголу или к существительному",
        ambiguous_fragment="с биноклем",
        alternative_attachment="Оборот 'с биноклем' может относиться к 'видел' (способ действия) или к 'человека' (атрибут человека).",
        explanation="Пояснение...",
        confidence=0.9,
        suggested_revision="Я видел в бинокль человека.",
        ambiguous_token_idx=5,
        alternative_parent_idx=2,
        alternative_relation="obl"
    )
    mock_response = AISyntaxResponse(
        is_ambiguous=True,
        findings=[mock_finding]
    )

    # 2. Setup document with linguistic analysis
    tokens = [
        Token(text="Я", start_char=0, end_char=1, is_punctuation=False, syntax_parent_idx=1, syntax_relation="nsubj"),
        Token(text="видел", start_char=2, end_char=7, is_punctuation=False, syntax_parent_idx=None, syntax_relation="root"),
        Token(text="человека", start_char=8, end_char=16, is_punctuation=False, syntax_parent_idx=1, syntax_relation="obj"),
        Token(text="с", start_char=17, end_char=18, is_punctuation=False, syntax_parent_idx=4, syntax_relation="case"),
        Token(text="биноклем", start_char=19, end_char=27, is_punctuation=False, syntax_parent_idx=2, syntax_relation="nmod"),
    ]
    
    sentence = Sentence(
        text="Я видел человека с биноклем",
        start_char=0,
        end_char=27,
        tokens=tokens
    )
    
    analysis = LinguisticAnalysis(
        document_id="doc1",
        sentence_count=1,
        token_count=5,
        sentences=[sentence]
    )
    
    doc = Document(
        id="doc1",
        title="essay.txt",
        raw_text="Я видел человека с биноклем",
        metadata=DocumentMetadata(
            file_name="essay.txt",
            file_type=SourceFileType.TXT,
            document_type=DocumentType.OTHER,
            parser_name="plain_text_parser",
        ),
        linguistic_analysis=analysis
    )

    detector = SyntaxAmbiguityDetector()
    
    # Mock OpenRouterClient.analyze_sentence_syntax
    with patch.object(detector._client, "analyze_sentence_syntax", return_value=mock_response) as mock_analyze:
        findings = detector.detect(doc)
        
        # Verify OpenRouterClient was called with sentence text
        mock_analyze.assert_called_once()
        args, kwargs = mock_analyze.call_args
        assert args[0] == "Я видел человека с биноклем"
        
        # Verify findings format
        assert len(findings) == 1
        f = findings[0]
        assert f.category == AmbiguityCategory.SYNTACTIC_AMBIGUITY
        assert f.source == FindingSource.LLM_AGENT
        assert f.detector_id == "syntax_ambiguity_detector"
        assert f.confidence == 0.9
        assert f.title == "Синтаксическая неоднозначность привязки предложной группы"
        assert f.span.text == "с биноклем"
        assert f.span.start_char == 17
        assert f.span.end_char == 27
        
        # Verify evidence features (AST override information)
        assert len(f.evidence) == 1
        ev = f.evidence[0]
        assert ev.features["ambiguous_token_idx"] == 5
        assert ev.features["alternative_parent_idx"] == 2
        assert ev.features["alternative_relation"] == "obl"


def test_syntax_ambiguity_detector_records_api_errors_as_warnings() -> None:
    mock_response = AISyntaxResponse(
        is_ambiguous=False,
        findings=[],
        error_message="Превышен лимит запросов OpenRouter (HTTP 429 Too Many Requests)."
    )

    tokens = [
        Token(text="Я", start_char=0, end_char=1, is_punctuation=False, syntax_parent_idx=1, syntax_relation="nsubj"),
        Token(text="видел", start_char=2, end_char=7, is_punctuation=False, syntax_parent_idx=None, syntax_relation="root"),
    ]
    
    sentence = Sentence(text="Я видел", start_char=0, end_char=7, tokens=tokens)
    analysis = LinguisticAnalysis(document_id="doc2", sentence_count=1, token_count=2, sentences=[sentence])
    doc = Document(
        id="doc2",
        title="essay.txt",
        raw_text="Я видел",
        metadata=DocumentMetadata(
            file_name="essay.txt",
            file_type=SourceFileType.TXT,
            document_type=DocumentType.OTHER,
            parser_name="plain_text_parser",
        ),
        linguistic_analysis=analysis
    )

    detector = SyntaxAmbiguityDetector()
    
    with patch.object(detector._client, "analyze_sentence_syntax", return_value=mock_response):
        findings = detector.detect(doc)
        
        assert len(findings) == 0
        assert len(doc.metadata.warnings) == 1
        assert "Превышен лимит запросов OpenRouter" in doc.metadata.warnings[0]
        assert "[Детектор синтаксической неоднозначности]" in doc.metadata.warnings[0]

from unittest.mock import patch
from hse_text_analyzer.domain.models.document import Document, DocumentMetadata
from hse_text_analyzer.domain.enums import SourceFileType, DocumentType, AmbiguityCategory, FindingSource
from hse_text_analyzer.domain.models.linguistic import LinguisticAnalysis, Sentence, Token
from hse_text_analyzer.plugins.detectors.referential_detector import ReferentialAmbiguityDetector
from hse_text_analyzer.infrastructure.ai.schemas import AIReferentialResponse, AIReferentialFinding


def test_referential_ambiguity_detector_creates_findings_when_ambiguous() -> None:
    # 1. Setup mock response from OpenRouter
    mock_finding = AIReferentialFinding(
        title="Неявная референциальная ссылка для местоимения 'она'",
        description="Местоимение 'она' может относиться к 'Мария' или к 'Анне'",
        ambiguous_fragment="она",
        possible_referents=["Мария", "Анне"],
        explanation="Пояснение...",
        confidence=0.85,
        suggested_revision="Мария сказала Анне о своем опоздании."
    )
    mock_response = AIReferentialResponse(
        is_ambiguous=True,
        findings=[mock_finding]
    )

    # 2. Setup document with linguistic analysis
    tokens = [
        Token(text="Мария", start_char=0, end_char=5, is_punctuation=False),
        Token(text="сказала", start_char=6, end_char=13, is_punctuation=False),
        Token(text="Анне", start_char=14, end_char=18, is_punctuation=False),
        Token(text="что", start_char=20, end_char=23, is_punctuation=False),
        Token(text="она", start_char=24, end_char=27, is_punctuation=False),
        Token(text="опоздает", start_char=28, end_char=36, is_punctuation=False),
    ]
    
    tokens[0].morphology.pos = "NOUN"
    tokens[0].morphology.number = "Sing"
    tokens[0].morphology.gender = "Fem"
    tokens[0].morphology.lemma = "мария"
    
    tokens[2].morphology.pos = "NOUN"
    tokens[2].morphology.number = "Sing"
    tokens[2].morphology.gender = "Fem"
    tokens[2].morphology.lemma = "анна"
    
    tokens[4].morphology.pos = "PRON"
    tokens[4].morphology.number = "Sing"
    tokens[4].morphology.gender = "Fem"
    tokens[4].morphology.lemma = "она"
    
    sentence = Sentence(
        text="Мария сказала Анне, что она опоздает",
        start_char=0,
        end_char=36,
        tokens=tokens
    )
    
    analysis = LinguisticAnalysis(
        document_id="doc3",
        sentence_count=1,
        token_count=6,
        sentences=[sentence]
    )
    
    doc = Document(
        id="doc3",
        title="essay.txt",
        raw_text="Мария сказала Анне, что она опоздает",
        metadata=DocumentMetadata(
            file_name="essay.txt",
            file_type=SourceFileType.TXT,
            document_type=DocumentType.OTHER,
            parser_name="plain_text_parser",
        ),
        linguistic_analysis=analysis
    )

    detector = ReferentialAmbiguityDetector()
    
    # Mock OpenRouterClient.analyze_sentence_reference
    with patch.object(detector._client, "analyze_sentence_reference", return_value=mock_response) as mock_analyze:
        findings = detector.detect(doc)
        
        # Verify OpenRouterClient was called with sentence text
        mock_analyze.assert_called_once()
        args, kwargs = mock_analyze.call_args
        assert args[0] == "Мария сказала Анне, что она опоздает"
        
        # Verify findings format
        assert len(findings) == 1
        f = findings[0]
        assert f.category == AmbiguityCategory.REFERENTIAL_AMBIGUITY
        assert f.source == FindingSource.LLM_AGENT
        assert f.detector_id == "referential_ambiguity_detector"
        assert f.confidence == 0.85
        assert f.title == "Неявная референциальная ссылка для местоимения 'она'"
        assert f.span.text == "она"
        assert f.span.start_char == 24
        assert f.span.end_char == 27
        
        assert len(f.evidence) == 1
        ev = f.evidence[0]
        assert ev.features["possible_referents"] == "Мария, Анне"


def test_referential_ambiguity_detector_skips_when_no_ambiguous_candidates() -> None:
    # Setup document where 'она' (Fem, Sing) only has 1 feminine singular noun before it ('Мария')
    # and 1 masculine singular noun ('Петру')
    tokens = [
        Token(text="Мария", start_char=0, end_char=5, is_punctuation=False),
        Token(text="сказала", start_char=6, end_char=13, is_punctuation=False),
        Token(text="Петру", start_char=14, end_char=19, is_punctuation=False),
        Token(text="что", start_char=21, end_char=24, is_punctuation=False),
        Token(text="она", start_char=25, end_char=28, is_punctuation=False),
        Token(text="опоздает", start_char=29, end_char=37, is_punctuation=False),
    ]
    
    tokens[0].morphology.pos = "NOUN"
    tokens[0].morphology.number = "Sing"
    tokens[0].morphology.gender = "Fem"
    tokens[0].morphology.lemma = "мария"
    
    tokens[2].morphology.pos = "NOUN"
    tokens[2].morphology.number = "Sing"
    tokens[2].morphology.gender = "Masc"
    tokens[2].morphology.lemma = "петр"
    
    tokens[4].morphology.pos = "PRON"
    tokens[4].morphology.number = "Sing"
    tokens[4].morphology.gender = "Fem"
    tokens[4].morphology.lemma = "она"
    
    sentence = Sentence(
        text="Мария сказала Петру, что она опоздает",
        start_char=0,
        end_char=37,
        tokens=tokens
    )
    
    analysis = LinguisticAnalysis(
        document_id="doc_pref1",
        sentence_count=1,
        token_count=6,
        sentences=[sentence]
    )
    
    doc = Document(
        id="doc_pref1",
        title="essay.txt",
        raw_text="Мария сказала Петру, что она опоздает",
        metadata=DocumentMetadata(
            file_name="essay.txt",
            file_type=SourceFileType.TXT,
            document_type=DocumentType.OTHER,
            parser_name="plain_text_parser",
        ),
        linguistic_analysis=analysis
    )

    detector = ReferentialAmbiguityDetector()
    
    with patch.object(detector._client, "analyze_sentence_reference") as mock_analyze:
        findings = detector.detect(doc)
        
        assert len(findings) == 0
        mock_analyze.assert_not_called()

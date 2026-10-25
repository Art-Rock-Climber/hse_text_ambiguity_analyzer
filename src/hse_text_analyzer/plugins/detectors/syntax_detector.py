import logging

from hse_text_analyzer.application.ports.ambiguity_detector import AmbiguityDetector, DetectorMetadata
from hse_text_analyzer.domain.enums import AmbiguityCategory, FindingSource, Severity, DocumentType
from hse_text_analyzer.domain.models.ambiguity import AmbiguityFinding, TextSpan, Evidence
from hse_text_analyzer.domain.models.document import Document
from hse_text_analyzer.infrastructure.ai.factory import get_ai_client
from hse_text_analyzer.infrastructure.ai.schemas import AISyntaxResponse, AISyntaxFinding
from hse_text_analyzer.infrastructure.rag.example_bank import ExampleBank
from hse_text_analyzer.plugins.detectors.detector_utils import (
    MAX_SENTENCES_PER_DOC,
    MAX_TOKENS_PER_SENTENCE,
    format_sentence_structure,
    find_fragment_span,
)

logger = logging.getLogger(__name__)


class SyntaxAmbiguityDetector(AmbiguityDetector):
    metadata = DetectorMetadata(
        detector_id="syntax_ambiguity_detector",
        name="Детектор синтаксической неоднозначности",
        description="Анализирует синтаксическую структуру предложений для выявления альтернативных вариантов связи слов",
        version="1.0.0",
        categories=(AmbiguityCategory.SYNTACTIC_AMBIGUITY,),
        supported_document_types=(DocumentType.OTHER,),
        default_enabled=True,
    )

    def __init__(self) -> None:
        self._client = get_ai_client()
        self._example_bank = ExampleBank()

    def detect(self, document: Document) -> list[AmbiguityFinding]:
        """
        Processes document sentences, formatting structure for LLM and retrieving findings.
        Applies budgets (max 20 sentences per doc) and skips long sentences (max 80 tokens).
        """
        findings: list[AmbiguityFinding] = []
        if not document.linguistic_analysis or not document.linguistic_analysis.sentences:
            return findings

        sentences_to_process = document.linguistic_analysis.sentences[:MAX_SENTENCES_PER_DOC]
        if len(document.linguistic_analysis.sentences) > MAX_SENTENCES_PER_DOC:
            logger.info(
                f"Budget check: document has {len(document.linguistic_analysis.sentences)} sentences. "
                f"Processing first {MAX_SENTENCES_PER_DOC}."
            )

        for sent_idx, sentence in enumerate(sentences_to_process):
            if not sentence.text or not sentence.text.strip():
                continue

            if len(sentence.tokens) > MAX_TOKENS_PER_SENTENCE:
                logger.info(
                    f"Limit check: Sentence #{sent_idx + 1} has {len(sentence.tokens)} tokens. Skipping."
                )
                continue

            structure_text = format_sentence_structure(sentence)

            # RAG: retrieve similar calibration examples and inject into prompt
            examples = self._example_bank.retrieve(sentence.text, "syntactic_ambiguity")
            few_shot_str = self._example_bank.format_for_prompt(examples)

            response = self._client.analyze_sentence_syntax(
                sentence.text, structure_text, few_shot_examples=few_shot_str
            )

            if response.error_message:
                warning_msg = f"[{self.metadata.name}] {response.error_message}"
                if warning_msg not in document.metadata.warnings:
                    document.metadata.warnings.append(warning_msg)
                break

            if response.is_ambiguous and response.findings:
                for f in response.findings:
                    span_start, span_end, frag = find_fragment_span(
                        sentence.text, f.ambiguous_fragment, sentence.start_char
                    )

                    span = TextSpan(
                        document_id=document.id,
                        start_char=span_start,
                        end_char=span_end,
                        text=frag,
                    )

                    # Store syntax override parameters in features dict of Evidence
                    features = {}
                    if f.ambiguous_token_idx is not None:
                        features["ambiguous_token_idx"] = int(f.ambiguous_token_idx)
                    if f.alternative_parent_idx is not None:
                        features["alternative_parent_idx"] = int(f.alternative_parent_idx)
                    if f.alternative_relation:
                        features["alternative_relation"] = str(f.alternative_relation)

                    evidence = Evidence(
                        message=f.alternative_attachment,
                        span=span,
                        features=features,
                    )

                    finding = AmbiguityFinding(
                        span=span,
                        category=AmbiguityCategory.SYNTACTIC_AMBIGUITY,
                        source=FindingSource.LLM_AGENT,
                        detector_id=self.metadata.detector_id,
                        severity=Severity.MEDIUM,
                        confidence=f.confidence,
                        title=f.title,
                        description=f.description,
                        evidence=[evidence],
                        suggested_revision=f.suggested_revision,
                        explanation=f.explanation,
                    )
                    findings.append(finding)

        return findings

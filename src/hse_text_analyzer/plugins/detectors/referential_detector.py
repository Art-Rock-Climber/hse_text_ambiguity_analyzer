import logging

from hse_text_analyzer.application.ports.ambiguity_detector import AmbiguityDetector, DetectorMetadata
from hse_text_analyzer.domain.enums import AmbiguityCategory, FindingSource, Severity, DocumentType
from hse_text_analyzer.domain.models.ambiguity import AmbiguityFinding, TextSpan, Evidence
from hse_text_analyzer.domain.models.document import Document
from hse_text_analyzer.infrastructure.ai.factory import get_ai_client
from hse_text_analyzer.infrastructure.ai.schemas import AIReferentialResponse, AIReferentialFinding
from hse_text_analyzer.plugins.detectors.detector_utils import (
    MAX_SENTENCES_PER_DOC,
    MAX_TOKENS_PER_SENTENCE,
    format_sentence_structure,
    find_fragment_span,
)

logger = logging.getLogger(__name__)


class ReferentialAmbiguityDetector(AmbiguityDetector):
    metadata = DetectorMetadata(
        detector_id="referential_ambiguity_detector",
        name="Детектор референциальной неоднозначности",
        description="Анализирует местоимения и указатели на предмет двусмысленности их соотнесения с существительными",
        version="1.0.0",
        categories=(AmbiguityCategory.REFERENTIAL_AMBIGUITY,),
        supported_document_types=(DocumentType.OTHER,),
        default_enabled=True,
    )

    _PRONOUN_LEMMAS = frozenset({
        "он", "она", "оно", "они", "его", "ее", "её", "их",
        "этот", "тот", "такой", "который",
    })

    def __init__(self) -> None:
        self._client = get_ai_client()

    def detect(self, document: Document) -> list[AmbiguityFinding]:
        """
        Detects pronoun/reference ambiguities using the AI client.
        Respects limitations check (max 20 sentences, max 80 tokens).
        """
        findings: list[AmbiguityFinding] = []
        if not document.linguistic_analysis or not document.linguistic_analysis.sentences:
            return findings

        sentences_to_process = document.linguistic_analysis.sentences[:MAX_SENTENCES_PER_DOC]

        for sent_idx, sentence in enumerate(sentences_to_process):
            if not sentence.text or not sentence.text.strip():
                continue

            if len(sentence.tokens) > MAX_TOKENS_PER_SENTENCE:
                continue

            # Prefiltering: skip unless there are 2+ antecedent candidates for a pronoun
            pronoun_tokens = [
                (idx, t) for idx, t in enumerate(sentence.tokens)
                if t.morphology
                and t.morphology.pos in {"PRON", "DET"}
                and (t.morphology.lemma or "").lower() in self._PRONOUN_LEMMAS
            ]

            has_potential_ambiguity = False
            for p_idx, p_tok in pronoun_tokens:
                p_num = p_tok.morphology.number
                p_gen = p_tok.morphology.gender

                candidates = [
                    t for idx, t in enumerate(sentence.tokens)
                    if idx < p_idx and t.morphology and t.morphology.pos in {"NOUN", "PROPN"}
                ]

                matching_candidates = []
                for c in candidates:
                    c_num = c.morphology.number
                    c_gen = c.morphology.gender

                    if p_num == "Plur":
                        if c_num == "Plur":
                            matching_candidates.append(c)
                    elif p_num == "Sing":
                        if c_num == "Sing":
                            if p_gen and c_gen:
                                if p_gen == c_gen:
                                    matching_candidates.append(c)
                            else:
                                matching_candidates.append(c)
                    else:
                        matching_candidates.append(c)

                if len(matching_candidates) >= 2:
                    has_potential_ambiguity = True
                    break

            if not has_potential_ambiguity:
                continue

            structure_text = format_sentence_structure(sentence)
            response = self._client.analyze_sentence_reference(sentence.text, structure_text)

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

                    features = {"possible_referents": ", ".join(f.possible_referents)}
                    evidence = Evidence(
                        message=f"Possible referents: {', '.join(f.possible_referents)}",
                        span=span,
                        features=features,
                    )

                    finding = AmbiguityFinding(
                        span=span,
                        category=AmbiguityCategory.REFERENTIAL_AMBIGUITY,
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

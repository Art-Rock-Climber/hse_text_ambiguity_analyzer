import logging
from hse_text_analyzer.application.ports.ambiguity_detector import AmbiguityDetector, DetectorMetadata
from hse_text_analyzer.domain.enums import AmbiguityCategory, FindingSource, Severity, DocumentType
from hse_text_analyzer.domain.models.ambiguity import AmbiguityFinding, TextSpan, Evidence
from hse_text_analyzer.domain.models.document import Document
from hse_text_analyzer.infrastructure.ai.factory import get_ai_client
from hse_text_analyzer.infrastructure.ai.schemas import AICoordinationalResponse, AICoordinationalFinding

logger = logging.getLogger(__name__)


class CoordinationalAmbiguityDetector(AmbiguityDetector):
    metadata = DetectorMetadata(
        detector_id="coordinational_ambiguity_detector",
        name="Детектор координационной неоднозначности",
        description="Анализирует группы элементов с союзами 'и/или' на предмет неоднозначности группировки",
        version="1.0.0",
        categories=(AmbiguityCategory.COORDINATIONAL_AMBIGUITY,),
        supported_document_types=(DocumentType.OTHER,),
        default_enabled=True,
    )

    def __init__(self) -> None:
        self._client = get_ai_client()

    def detect(self, document: Document) -> list[AmbiguityFinding]:
        """
        Detects coordinational grouping ambiguities using the OpenRouter AI client.
        Respects limitations check (max 20 sentences, max 80 tokens).
        """
        findings: list[AmbiguityFinding] = []
        if not document.linguistic_analysis or not document.linguistic_analysis.sentences:
            return findings

        # Limitations budget check: Max 20 sentences per document
        max_sentences = 20
        sentences_to_process = document.linguistic_analysis.sentences[:max_sentences]

        for sent_idx, sentence in enumerate(sentences_to_process):
            if not sentence.text or not sentence.text.strip():
                continue
                
            if len(sentence.tokens) > 80:
                continue

            # Prefiltering optimization check
            cc_tokens = [t for t in sentence.tokens if t.text.lower() in {"и", "или", "либо"}]
            comma_tokens = [t for t in sentence.tokens if t.text == ","]
            
            cc_count = len(cc_tokens)
            comma_count = len(comma_tokens)
            
            has_i = any(t.text.lower() == "и" for t in cc_tokens)
            has_ili = any(t.text.lower() == "или" for t in cc_tokens)
            has_libo = any(t.text.lower() == "либо" for t in cc_tokens)
            conjunction_types = sum([has_i, has_ili, has_libo])
            
            should_call = (
                cc_count >= 2 or
                conjunction_types >= 2 or
                comma_count >= 2 or
                (comma_count >= 1 and cc_count >= 1)
            )
            
            if not should_call:
                continue

            # Format sentence tokens & syntax dependency tree into readable representation
            structure_lines = []
            for t_idx, t in enumerate(sentence.tokens):
                parent_idx = t.syntax_parent_idx
                parent_info = "ROOT"
                if parent_idx is not None and 0 <= parent_idx < len(sentence.tokens):
                    parent_token = sentence.tokens[parent_idx]
                    parent_info = f"«{parent_token.text}» (# {parent_idx + 1})"

                morph_list = []
                m = t.morphology
                if m.pos: morph_list.append(f"POS: {m.pos}")
                if m.lemma: morph_list.append(f"lemma: {m.lemma}")
                if m.gender: morph_list.append(f"gender: {m.gender}")
                if m.number: morph_list.append(f"number: {m.number}")
                if m.case: morph_list.append(f"case: {m.case}")
                
                morph_str = ", ".join(morph_list)
                structure_lines.append(
                    f"{t_idx + 1}. {t.text} ({morph_str}) -> parent: {parent_info}, relation: {t.syntax_relation or 'None'}"
                )

            structure_text = "\n".join(structure_lines)
            
            # Query client
            response = self._client.analyze_sentence_coordination(sentence.text, structure_text)
            
            if response.error_message:
                warning_msg = f"[{self.metadata.name}] {response.error_message}"
                if warning_msg not in document.metadata.warnings:
                    document.metadata.warnings.append(warning_msg)
                break

            if response.is_ambiguous and response.findings:
                for f in response.findings:
                    frag = f.ambiguous_fragment
                    start_pos = sentence.text.find(frag)
                    if start_pos != -1:
                        span_start = sentence.start_char + start_pos
                        span_end = span_start + len(frag)
                    else:
                        span_start = sentence.start_char
                        span_end = sentence.end_char
                        frag = sentence.text

                    span = TextSpan(
                        document_id=document.id,
                        start_char=span_start,
                        end_char=span_end,
                        text=frag,
                    )
                    
                    evidence = Evidence(
                        message=f"Ambiguous coordinate group: {frag}",
                        span=span,
                    )

                    finding = AmbiguityFinding(
                        span=span,
                        category=AmbiguityCategory.COORDINATIONAL_AMBIGUITY,
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

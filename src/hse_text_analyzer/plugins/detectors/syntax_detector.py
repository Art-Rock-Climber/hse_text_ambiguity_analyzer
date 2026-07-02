import logging
from hse_text_analyzer.application.ports.ambiguity_detector import AmbiguityDetector, DetectorMetadata
from hse_text_analyzer.domain.enums import AmbiguityCategory, FindingSource, Severity, DocumentType
from hse_text_analyzer.domain.models.ambiguity import AmbiguityFinding, TextSpan, Evidence
from hse_text_analyzer.domain.models.document import Document
from hse_text_analyzer.infrastructure.ai.factory import get_ai_client
from hse_text_analyzer.infrastructure.ai.schemas import AISyntaxResponse, AISyntaxFinding

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

    def detect(self, document: Document) -> list[AmbiguityFinding]:
        """
        Processes document sentences, formatting structure for LLM and retrieving findings.
        Applies budgets (max 20 sentences per doc) and skips long sentences (max 80 tokens).
        """
        findings: list[AmbiguityFinding] = []
        if not document.linguistic_analysis or not document.linguistic_analysis.sentences:
            return findings

        # Limitations budget check: Max 20 sentences per document
        max_sentences = 20
        sentences_to_process = document.linguistic_analysis.sentences[:max_sentences]
        if len(document.linguistic_analysis.sentences) > max_sentences:
            logger.info(f"Budget check: document has {len(document.linguistic_analysis.sentences)} sentences. Processing first {max_sentences}.")

        for sent_idx, sentence in enumerate(sentences_to_process):
            # Skip empty or trivially short sentences
            if not sentence.text or not sentence.text.strip():
                continue
                
            # Limit check: Skip sentence if it has more than 80 tokens
            if len(sentence.tokens) > 80:
                logger.info(f"Limit check: Sentence #{sent_idx + 1} has {len(sentence.tokens)} tokens. Skipping.")
                continue

            # Format sentence tokens & syntax dependency tree into readable representation
            structure_lines = []
            for t_idx, t in enumerate(sentence.tokens):
                parent_idx = t.syntax_parent_idx
                parent_info = "ROOT"
                if parent_idx is not None and 0 <= parent_idx < len(sentence.tokens):
                    parent_token = sentence.tokens[parent_idx]
                    parent_info = f"«{parent_token.text}» (#{parent_idx + 1})"

                morph_list = []
                m = t.morphology
                if m.pos: morph_list.append(f"POS: {m.pos}")
                if m.lemma: morph_list.append(f"lemma: {m.lemma}")
                if m.gender: morph_list.append(f"gender: {m.gender}")
                if m.number: morph_list.append(f"number: {m.number}")
                if m.case: morph_list.append(f"case: {m.case}")
                if m.tense: morph_list.append(f"tense: {m.tense}")
                if m.aspect: morph_list.append(f"aspect: {m.aspect}")
                if m.person: morph_list.append(f"person: {m.person}")
                
                morph_str = ", ".join(morph_list)
                structure_lines.append(
                    f"{t_idx + 1}. {t.text} ({morph_str}) -> parent: {parent_info}, relation: {t.syntax_relation or 'None'}"
                )

            structure_text = "\n".join(structure_lines)
            
            # Query client
            response = self._client.analyze_sentence_syntax(sentence.text, structure_text)
            
            if response.error_message:
                warning_msg = f"[{self.metadata.name}] {response.error_message}"
                if warning_msg not in document.metadata.warnings:
                    document.metadata.warnings.append(warning_msg)
                break

            if response.is_ambiguous and response.findings:
                for f in response.findings:
                    # Resolve start/end characters of ambiguous fragment inside the sentence
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

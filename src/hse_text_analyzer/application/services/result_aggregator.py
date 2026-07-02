from hse_text_analyzer.domain.enums import FindingSource
from hse_text_analyzer.domain.models.ambiguity import AmbiguityFinding
from hse_text_analyzer.domain.models.analysis import AnalysisResult, AnalysisSummary
from hse_text_analyzer.domain.models.document import Document


class ResultAggregator:
    def aggregate(self, document: Document, findings: list[AmbiguityFinding]) -> AnalysisResult:
        return AnalysisResult(
            document_id=document.id,
            findings=findings,
            summary=AnalysisSummary(
                total_findings=len(findings),
                rule_based_findings=sum(
                    1 for finding in findings if finding.source == FindingSource.RULE_BASED
                ),
                ai_findings=sum(
                    1 for finding in findings if finding.source == FindingSource.LLM_AGENT
                ),
                parser_warnings=document.metadata.warnings,
            ),
        )

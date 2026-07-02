from enum import StrEnum


class DocumentType(StrEnum):
    ASSIGNMENT = "assignment"
    LAB_WORK = "lab_work"
    REPORT = "report"
    ESSAY = "essay"
    PROJECT_DOCUMENTATION = "project_documentation"
    REQUIREMENTS_SPECIFICATION = "requirements_specification"
    OTHER = "other"


class SourceFileType(StrEnum):
    TXT = "txt"
    MARKDOWN = "md"
    DOCX = "docx"
    PDF = "pdf"


class AmbiguityCategory(StrEnum):
    LEXICAL_VAGUENESS = "lexical_vagueness"
    REFERENTIAL_AMBIGUITY = "referential_ambiguity"
    SYNTACTIC_AMBIGUITY = "syntactic_ambiguity"
    COORDINATIONAL_AMBIGUITY = "coordinational_ambiguity"
    UNDERSPECIFIED_REQUIREMENT = "underspecified_requirement"
    UNCLEAR_INSTRUCTION = "unclear_instruction"
    MISSING_CONTEXT = "missing_context"
    MISSING_REFERENCE = "missing_reference"
    AMBIGUOUS_QUANTIFIER = "ambiguous_quantifier"
    INCONSISTENT_TERMINOLOGY = "inconsistent_terminology"
    METHODOLOGICAL_AMBIGUITY = "methodological_ambiguity"
    CONTRADICTORY_REQUIREMENT = "contradictory_requirement"


class FindingSource(StrEnum):
    RULE_BASED = "rule_based"
    LLM_AGENT = "llm_agent"
    GUARDIAN_VALIDATED = "guardian_validated"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

"""
Shared utilities for ambiguity detectors.
Centralises the sentence-budget constants and sentence structure
formatting that was previously copy-pasted into every detector.
"""
from __future__ import annotations

from hse_text_analyzer.domain.models.linguistic import Sentence

# Hard limits applied by every detector uniformly
MAX_SENTENCES_PER_DOC = 20
MAX_TOKENS_PER_SENTENCE = 80


def format_sentence_structure(sentence: Sentence) -> str:
    """
    Convert a Sentence's token list and dependency tree into a numbered
    string suitable for inclusion in an LLM prompt.

    Example output line:
        3. студента (POS: NOUN, lemma: студент, case: Gent) -> parent: «видел» (#2), relation: obj
    """
    lines: list[str] = []
    for t_idx, t in enumerate(sentence.tokens):
        parent_idx = t.syntax_parent_idx
        if parent_idx is not None and 0 <= parent_idx < len(sentence.tokens):
            parent_token = sentence.tokens[parent_idx]
            parent_info = f"«{parent_token.text}» (#{parent_idx + 1})"
        else:
            parent_info = "ROOT"

        morph_list: list[str] = []
        m = t.morphology
        if m.pos:    morph_list.append(f"POS: {m.pos}")        # noqa: E701
        if m.lemma:  morph_list.append(f"lemma: {m.lemma}")    # noqa: E701
        if m.gender: morph_list.append(f"gender: {m.gender}")  # noqa: E701
        if m.number: morph_list.append(f"number: {m.number}")  # noqa: E701
        if m.case:   morph_list.append(f"case: {m.case}")      # noqa: E701
        if m.tense:  morph_list.append(f"tense: {m.tense}")    # noqa: E701
        if m.aspect: morph_list.append(f"aspect: {m.aspect}")  # noqa: E701
        if m.person: morph_list.append(f"person: {m.person}")  # noqa: E701

        morph_str = ", ".join(morph_list)
        lines.append(
            f"{t_idx + 1}. {t.text} ({morph_str}) -> parent: {parent_info},"
            f" relation: {t.syntax_relation or 'None'}"
        )

    return "\n".join(lines)


def find_fragment_span(sentence_text: str, fragment: str, sentence_start_char: int) -> tuple[int, int, str]:
    """
    Locate *fragment* inside *sentence_text* and return absolute document
    character offsets (start_char, end_char, resolved_text).

    Falls back to the whole sentence if the fragment is not found, which
    is safer than silently using the wrong position.
    """
    start_pos = sentence_text.find(fragment)
    if start_pos != -1:
        abs_start = sentence_start_char + start_pos
        abs_end = abs_start + len(fragment)
        return abs_start, abs_end, fragment

    # Fragment not found — fall back to the full sentence span
    return sentence_start_char, sentence_start_char + len(sentence_text), sentence_text

import re


_SENTENCE_END_RE = re.compile(r"[.!?…]+(?=\s|$)", re.UNICODE)
_TOKEN_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)*|[^\w\s]",
    re.UNICODE,
)


def segment_text_into_sentences(text: str) -> list[dict[str, int | str]]:
    sentences: list[dict[str, int | str]] = []
    if not text:
        return sentences

    current_offset = 0
    for match in _SENTENCE_END_RE.finditer(text):
        end_offset = match.end()
        raw_segment = text[current_offset:end_offset]
        trimmed = raw_segment.strip()
        if trimmed:
            leading = len(raw_segment) - len(raw_segment.lstrip())
            start = current_offset + leading
            sentences.append(
                {
                    "text": trimmed,
                    "start_char": start,
                    "end_char": start + len(trimmed),
                }
            )
        current_offset = end_offset

    if current_offset < len(text):
        raw_segment = text[current_offset:]
        trimmed = raw_segment.strip()
        if trimmed:
            leading = len(raw_segment) - len(raw_segment.lstrip())
            start = current_offset + leading
            sentences.append(
                {
                    "text": trimmed,
                    "start_char": start,
                    "end_char": start + len(trimmed),
                }
            )

    return sentences


def tokenize_text(text: str) -> list[dict[str, int | str | bool]]:
    tokens: list[dict[str, int | str | bool]] = []
    for match in _TOKEN_RE.finditer(text):
        token_text = match.group(0)
        tokens.append(
            {
                "text": token_text,
                "start_char": match.start(),
                "end_char": match.end(),
                "is_punctuation": bool(re.fullmatch(r"[^\w\s]", token_text, re.UNICODE)),
            }
        )
    return tokens

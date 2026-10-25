"""
RAG (Retrieval-Augmented Generation) example bank for Russian ambiguity detection.

Maintains a curated set of labeled Russian sentences per ambiguity category.
Before each LLM call, the two most lexically similar examples are retrieved
and injected into the prompt as few-shot calibration examples.

Retrieval uses lemma-level Jaccard similarity via pymorphy3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RussianAmbiguityExample:
    """A single labeled sentence used for few-shot retrieval."""
    sentence: str
    is_ambiguous: bool
    explanation: str   # short human-readable rationale


# ---------------------------------------------------------------------------
# Curated example bank
# ---------------------------------------------------------------------------

_SYNTAX_EXAMPLES: list[RussianAmbiguityExample] = [
    RussianAmbiguityExample(
        sentence="Я видел человека с биноклем.",
        is_ambiguous=True,
        explanation="«с биноклем» может относиться как к «видел» (я смотрел в бинокль), "
                    "так и к «человека» (у человека был бинокль).",
    ),
    RussianAmbiguityExample(
        sentence="Профессор объяснил задачу студентам в аудитории.",
        is_ambiguous=True,
        explanation="«в аудитории» может относиться к «объяснил» (объяснение происходило в аудитории) "
                    "или к «студентам» (студенты находились в аудитории).",
    ),
    RussianAmbiguityExample(
        sentence="Он прочитал статью о войне на диване.",
        is_ambiguous=True,
        explanation="«на диване» может относиться к «прочитал» (читал лёжа на диване) "
                    "или к «статью» (статья о войне на диване — странно, но грамматически допустимо).",
    ),
    RussianAmbiguityExample(
        sentence="Студент написал курсовую работу.",
        is_ambiguous=False,
        explanation="Предложение однозначно: один субъект, один глагол, один объект — нет альтернативных синтаксических разборов.",
    ),
    RussianAmbiguityExample(
        sentence="Преподаватель проверил контрольную работу.",
        is_ambiguous=False,
        explanation="Нет двусмысленных привязок: структура Subject–Verb–Object полностью однозначна.",
    ),
    RussianAmbiguityExample(
        sentence="Декан подписал приказ об отчислении.",
        is_ambiguous=False,
        explanation="Предложение однозначно: «об отчислении» однозначно поясняет «приказ», других допустимых привязок нет.",
    ),
]

_REFERENTIAL_EXAMPLES: list[RussianAmbiguityExample] = [
    RussianAmbiguityExample(
        sentence="Мария сказала Анне, что она опоздает.",
        is_ambiguous=True,
        explanation="«она» может относиться как к Марии, так и к Анне — оба антецедента "
                    "согласуются по роду и числу.",
    ),
    RussianAmbiguityExample(
        sentence="Исследователь обсудил с коллегой его гипотезу.",
        is_ambiguous=True,
        explanation="«его» может относиться к «исследователю» или к «коллеге» — "
                    "оба существительные мужского рода.",
    ),
    RussianAmbiguityExample(
        sentence="Директор похвалил студента, потому что он хорошо выступил.",
        is_ambiguous=True,
        explanation="«он» может относиться как к «директору», так и к «студенту» — "
                    "оба мужского рода, оба упомянуты до местоимения.",
    ),
    RussianAmbiguityExample(
        sentence="Студент сдал экзамен, и он был рад.",
        is_ambiguous=False,
        explanation="«он» однозначно относится к «студенту» — единственное лицо мужского рода в предложении.",
    ),
    RussianAmbiguityExample(
        sentence="Компьютер завис, и его пришлось перезагрузить.",
        is_ambiguous=False,
        explanation="«его» однозначно относится к «компьютеру» — единственный антецедент.",
    ),
    RussianAmbiguityExample(
        sentence="Анна позвонила Марине и рассказала ей о встрече.",
        is_ambiguous=False,
        explanation="«ей» однозначно относится к «Марине» — Анна звонила Марине, поэтому "
                    "«рассказала ей» = рассказала Марине.",
    ),
]

_COORDINATIONAL_EXAMPLES: list[RussianAmbiguityExample] = [
    RussianAmbiguityExample(
        sentence="Студенты и преподаватели и родители пришли на встречу.",
        is_ambiguous=True,
        explanation="Два союза «и» создают неоднозначность: можно трактовать как "
                    "(студенты и преподаватели) и родители, или студенты и (преподаватели и родители).",
    ),
    RussianAmbiguityExample(
        sentence="Необходимо изучить алгоритмы и структуры данных и теорию вероятностей.",
        is_ambiguous=True,
        explanation="Неясно, являются ли «алгоритмы и структуры данных» одной группой, "
                    "или все три объекта перечислены независимо.",
    ),
    RussianAmbiguityExample(
        sentence="Отчёт проверили научный руководитель или консультант и заведующий кафедрой.",
        is_ambiguous=True,
        explanation="Смешение «или» и «и» делает неясным, кто именно проверял: "
                    "(руководитель или консультант) и завкафедрой, "
                    "или руководитель или (консультант и завкафедрой)?",
    ),
    RussianAmbiguityExample(
        sentence="Студенты принесли тетради и ручки.",
        is_ambiguous=False,
        explanation="Простое перечисление двух объектов с одним «и» — однозначная конъюнкция.",
    ),
    RussianAmbiguityExample(
        sentence="Можно сдать зачёт или экзамен.",
        is_ambiguous=False,
        explanation="Простая бинарная альтернатива с «или» — нет неоднозначной группировки.",
    ),
    RussianAmbiguityExample(
        sentence="Научный руководитель и декан подписали документ.",
        is_ambiguous=False,
        explanation="Два участника объединены одним «и» — однозначная конъюнкция субъектов.",
    ),
]

_BANK: dict[str, list[RussianAmbiguityExample]] = {
    "syntactic_ambiguity": _SYNTAX_EXAMPLES,
    "referential_ambiguity": _REFERENTIAL_EXAMPLES,
    "coordinational_ambiguity": _COORDINATIONAL_EXAMPLES,
}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _lemmatize(text: str) -> frozenset[str]:
    """
    Return the set of lemmas for all words in *text*, lowercased.
    Uses pymorphy3 (already a project dependency).
    Cached so repeated calls for the same text are free.
    """
    try:
        import pymorphy3  # type: ignore
        morph = pymorphy3.MorphAnalyzer()
        words = text.lower().split()
        return frozenset(
            morph.parse(w)[0].normal_form
            for w in words
            if w.isalpha()
        )
    except Exception:
        # Fallback: simple lowercase token set
        return frozenset(w.lower() for w in text.split() if w.isalpha())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class ExampleBank:
    """
    Retrieves the most relevant few-shot examples for a given sentence
    and ambiguity category using lemma-level Jaccard similarity.
    """

    def retrieve(
        self,
        sentence_text: str,
        category: str,
        n: int = 2,
    ) -> list[RussianAmbiguityExample]:
        """
        Return up to *n* examples from the bank for *category*, ranked by
        Jaccard similarity to *sentence_text*. One positive and one negative
        example are preferred to maximise calibration value.
        """
        examples = _BANK.get(category, [])
        if not examples:
            return []

        query_lemmas = _lemmatize(sentence_text)
        scored = sorted(
            examples,
            key=lambda ex: _jaccard(query_lemmas, _lemmatize(ex.sentence)),
            reverse=True,
        )

        # Try to return at least one positive and one negative example
        positives = [e for e in scored if e.is_ambiguous]
        negatives = [e for e in scored if not e.is_ambiguous]

        result: list[RussianAmbiguityExample] = []
        if positives:
            result.append(positives[0])
        if negatives:
            result.append(negatives[0])

        # Fill remaining slots from scored list (avoiding duplicates)
        for e in scored:
            if len(result) >= n:
                break
            if e not in result:
                result.append(e)

        return result[:n]

    def format_for_prompt(self, examples: list[RussianAmbiguityExample]) -> str:
        """
        Format retrieved examples as a compact few-shot block for inclusion
        in an LLM prompt.  Returns an empty string if *examples* is empty.
        """
        if not examples:
            return ""

        lines = ["Примеры для ориентира (не анализируй их — только используй как образец):"]
        for i, ex in enumerate(examples, 1):
            label = "НЕОДНОЗНАЧНО" if ex.is_ambiguous else "НЕ НЕОДНОЗНАЧНО"
            lines.append(
                f"\n  Пример {i}: «{ex.sentence}»\n"
                f"  Вердикт: {label}\n"
                f"  Пояснение: {ex.explanation}"
            )
        return "\n".join(lines)

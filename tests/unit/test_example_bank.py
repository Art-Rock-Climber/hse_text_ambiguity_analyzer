import pytest
from hse_text_analyzer.infrastructure.rag.example_bank import ExampleBank, RussianAmbiguityExample


@pytest.fixture
def bank() -> ExampleBank:
    return ExampleBank()


def test_retrieve_returns_correct_count(bank: ExampleBank) -> None:
    results = bank.retrieve("Студенты и преподаватели пришли.", "coordinational_ambiguity", n=2)
    assert len(results) <= 2


def test_retrieve_returns_both_positive_and_negative(bank: ExampleBank) -> None:
    results = bank.retrieve("Мария сказала Анне, что она придёт.", "referential_ambiguity", n=2)
    labels = {r.is_ambiguous for r in results}
    # Preferred behaviour: one positive and one negative example returned
    assert True in labels and False in labels


def test_retrieve_returns_empty_for_unknown_category(bank: ExampleBank) -> None:
    results = bank.retrieve("Любое предложение.", "unknown_category", n=2)
    assert results == []


def test_format_for_prompt_is_empty_for_no_examples(bank: ExampleBank) -> None:
    assert bank.format_for_prompt([]) == ""


def test_format_for_prompt_contains_sentence_text(bank: ExampleBank) -> None:
    examples = bank.retrieve("Он видел человека с биноклем.", "syntactic_ambiguity", n=2)
    formatted = bank.format_for_prompt(examples)
    assert "Пример" in formatted
    # At least one retrieved sentence should appear in the formatted output
    assert any(ex.sentence in formatted for ex in examples)


def test_retrieve_all_three_categories_are_populated(bank: ExampleBank) -> None:
    for category in ("syntactic_ambiguity", "referential_ambiguity", "coordinational_ambiguity"):
        results = bank.retrieve("тест предложение", category, n=1)
        assert len(results) >= 1, f"Category {category} returned no examples"

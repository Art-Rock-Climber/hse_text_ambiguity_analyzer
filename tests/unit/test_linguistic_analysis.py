from hse_text_analyzer.application.ports.document_parser import UploadedDocument
from hse_text_analyzer.application.services.composition import build_analysis_workflow


def test_russian_linguistic_analysis_attaches_sentence_and_token_data() -> None:
    workflow = build_analysis_workflow()
    uploaded = UploadedDocument(
        file_name="essay.txt",
        content="Это первый тест. Второе предложение!".encode(),
    )

    document, result = workflow.run(uploaded)

    assert document.linguistic_analysis is not None
    assert document.linguistic_analysis.sentence_count == 2
    assert document.linguistic_analysis.token_count >= 6
    assert result.document_id == document.id
    assert document.linguistic_analysis.sentences[0].text.startswith("Это первый")
    assert any(token.morphology.lemma for token in document.linguistic_analysis.sentences[0].tokens if not token.is_punctuation)

    # Check syntactic parsing data
    first_sentence = document.linguistic_analysis.sentences[0]
    assert any(token.syntax_relation is not None for token in first_sentence.tokens)
    assert any(token.syntax_parent_idx is not None for token in first_sentence.tokens)

    # Check detailed morphological categories (e.g. gender and number)
    second_sentence = document.linguistic_analysis.sentences[1]
    predlozhenie_token = next(t for t in second_sentence.tokens if t.text == "предложение")
    assert predlozhenie_token.morphology.number == "sing"
    assert predlozhenie_token.morphology.gender == "neut"

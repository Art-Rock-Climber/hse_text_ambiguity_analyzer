import pytest
from unittest.mock import patch
from hse_text_analyzer.infrastructure.ai.schemas import AISyntaxResponse, AIReferentialResponse, AICoordinationalResponse


@pytest.fixture(autouse=True)
def mock_openrouter_client():
    """
    Globally mocks BaseAIClient API methods to prevent real API calls
    during tests, returning empty responses by default.
    """
    empty_syntax = AISyntaxResponse(is_ambiguous=False, findings=[])
    empty_reference = AIReferentialResponse(is_ambiguous=False, findings=[])
    empty_coordination = AICoordinationalResponse(is_ambiguous=False, findings=[])
    
    with patch(
        "hse_text_analyzer.infrastructure.ai.base_client.BaseAIClient.analyze_sentence_syntax",
        return_value=empty_syntax,
    ) as mock_s, patch(
        "hse_text_analyzer.infrastructure.ai.base_client.BaseAIClient.analyze_sentence_reference",
        return_value=empty_reference,
    ) as mock_r, patch(
        "hse_text_analyzer.infrastructure.ai.base_client.BaseAIClient.analyze_sentence_coordination",
        return_value=empty_coordination,
    ) as mock_c:
        yield mock_s, mock_r, mock_c

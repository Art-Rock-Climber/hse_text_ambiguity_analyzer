import pytest
from unittest.mock import patch, MagicMock
import httpx
from hse_text_analyzer.infrastructure.ai.groq_client import GroqClient


def test_groq_client_succeeds_on_first_attempt() -> None:
    client_instance = GroqClient()
    client_instance.settings.groq_api_key = "dummy_key"
    client_instance.settings.groq_model = "llama3-70b-8192"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"is_ambiguous": false, "findings": []}'}}]
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        content, err = client_instance._post_with_retry("hello")

        assert err is None
        assert content == '{"is_ambiguous": false, "findings": []}'
        assert mock_post.call_count == 1
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "llama3-70b-8192"


def test_groq_client_retries_on_429_then_succeeds() -> None:
    client_instance = GroqClient()
    client_instance.settings.groq_api_key = "dummy_key"
    client_instance.settings.groq_model = "llama3-70b-8192"

    mock_429 = MagicMock()
    mock_429.status_code = 429

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "choices": [{"message": {"content": '{"is_ambiguous": false, "findings": []}'}}]
    }

    with patch("httpx.Client.post", side_effect=[mock_429, mock_200]) as mock_post, \
         patch("hse_text_analyzer.infrastructure.ai.groq_client.sleep") as mock_sleep:

        content, err = client_instance._post_with_retry("hello")

        assert err is None
        assert content == '{"is_ambiguous": false, "findings": []}'
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1.0)


def test_groq_client_exhausts_all_retries_and_returns_error() -> None:
    client_instance = GroqClient()
    client_instance.settings.groq_api_key = "dummy_key"
    client_instance.settings.groq_model = "llama3-70b-8192"

    mock_429 = MagicMock()
    mock_429.status_code = 429

    with patch("httpx.Client.post", return_value=mock_429) as mock_post, \
         patch("hse_text_analyzer.infrastructure.ai.groq_client.sleep") as mock_sleep:

        content, err = client_instance._post_with_retry("hello")

        assert content is None
        assert err is not None
        assert "429" in err or "Groq" in err
        # 3 attempts, sleep called after first and second (not after last)
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2


def test_groq_client_returns_error_on_non_429_http_error() -> None:
    client_instance = GroqClient()
    client_instance.settings.groq_api_key = "dummy_key"
    client_instance.settings.groq_model = "llama3-70b-8192"

    mock_500 = MagicMock()
    mock_500.status_code = 500
    mock_500.text = "Internal Server Error"

    with patch("httpx.Client.post", return_value=mock_500) as mock_post:
        content, err = client_instance._post_with_retry("hello")

        assert content is None
        assert err is not None
        assert "500" in err
        # Must NOT retry on non-429 errors
        assert mock_post.call_count == 1

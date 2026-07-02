import pytest
from unittest.mock import patch, MagicMock
import httpx
from hse_text_analyzer.infrastructure.ai.openrouter_client import OpenRouterClient


def test_openrouter_client_succeeds_on_first_attempt() -> None:
    client_instance = OpenRouterClient()
    client_instance.settings.openrouter_api_key = "dummy_key"
    client_instance.settings.openrouter_model = "primary-model"
    client_instance.settings.openrouter_model_extra_1 = "fallback-model"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"is_ambiguous": false, "findings": []}'
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        content, err = client_instance._post_with_retry("hello")
        
        assert err is None
        assert content == '{"is_ambiguous": false, "findings": []}'
        assert mock_post.call_count == 1
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "primary-model"


def test_openrouter_client_retries_on_429() -> None:
    client_instance = OpenRouterClient()
    client_instance.settings.openrouter_api_key = "dummy_key"
    client_instance.settings.openrouter_model = "primary-model"
    client_instance.settings.openrouter_model_extra_1 = None

    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429
    
    mock_response_200 = MagicMock()
    mock_response_200.status_code = 200
    mock_response_200.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"is_ambiguous": false, "findings": []}'
                }
            }
        ]
    }

    with patch("httpx.Client.post", side_effect=[mock_response_429, mock_response_200]) as mock_post, \
         patch("hse_text_analyzer.infrastructure.ai.openrouter_client.sleep") as mock_sleep:
         
        content, err = client_instance._post_with_retry("hello")
        
        assert err is None
        assert content == '{"is_ambiguous": false, "findings": []}'
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1.0)


def test_openrouter_client_falls_back_on_exhaustion() -> None:
    client_instance = OpenRouterClient()
    client_instance.settings.openrouter_api_key = "dummy_key"
    client_instance.settings.openrouter_model = "primary-model"
    client_instance.settings.openrouter_model_extra_1 = "fallback-model"

    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429
    
    mock_response_200 = MagicMock()
    mock_response_200.status_code = 200
    mock_response_200.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"is_ambiguous": false, "findings": []}'
                }
            }
        ]
    }

    # Primary model fails 3 times, fallback model succeeds on 1st attempt
    side_effects = [mock_response_429, mock_response_429, mock_response_429, mock_response_200]

    with patch("httpx.Client.post", side_effect=side_effects) as mock_post, \
         patch("hse_text_analyzer.infrastructure.ai.openrouter_client.sleep") as mock_sleep:
         
        content, err = client_instance._post_with_retry("hello")
        
        assert err is None
        assert content == '{"is_ambiguous": false, "findings": []}'
        assert mock_post.call_count == 4
        assert mock_sleep.call_count == 2
        
        calls = mock_post.call_args_list
        assert calls[0][1]["json"]["model"] == "primary-model"
        assert calls[1][1]["json"]["model"] == "primary-model"
        assert calls[2][1]["json"]["model"] == "primary-model"
        assert calls[3][1]["json"]["model"] == "fallback-model"

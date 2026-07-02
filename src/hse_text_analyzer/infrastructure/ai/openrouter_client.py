import logging
from time import sleep
import httpx

from hse_text_analyzer.infrastructure.ai.base_client import BaseAIClient
from hse_text_analyzer.configuration.settings import get_settings

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseAIClient):
    """
    OpenRouter implementation of the AI client.
    """
    def __init__(self) -> None:
        super().__init__(get_settings())

    @property
    def _api_key(self) -> str | None:
        return self.settings.openrouter_api_key

    @property
    def _provider_name(self) -> str:
        return "OpenRouter"

    def _post_with_retry(self, prompt: str) -> tuple[str | None, str | None]:
        """
        Sends requests to OpenRouter with a retry loop on 429 and fallback model support.
        Returns a tuple of (content, error_message).
        """
        models = [self.settings.openrouter_model]
        if self.settings.openrouter_model_extra_1:
            models.append(self.settings.openrouter_model_extra_1)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/hse-text-analyzer",
            "X-Title": "HSE Text Analyzer",
        }

        last_error_msg = None

        for model in models:
            for attempt in range(3):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,
                }
                
                try:
                    with httpx.Client(
                        timeout=httpx.Timeout(
                        connect=10.0,
                        read=60.0,
                        write=10.0,
                        pool=10.0,
                    )) as client:
                        response = client.post(
                            self.settings.openrouter_api_url,
                            headers=headers,
                            json=payload,
                        )
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            content = response_data["choices"][0]["message"]["content"]
                            return content, None
                            
                        logger.error(f"OpenRouter API error (Attempt {attempt+1}) for model {model}: Status {response.status_code}, Body: {response.text}")
                        
                        if response.status_code == 429:
                            last_error_msg = f"Превышен лимит запросов OpenRouter (HTTP 429 Too Many Requests) для модели {model}."
                            if attempt < 2:
                                sleep(1.0)
                            continue
                        
                        return None, f"Ошибка OpenRouter API (HTTP {response.status_code}) для модели {model}."
                        
                except httpx.HTTPError as exc:
                    logger.error(f"HTTP error connecting to OpenRouter for model {model} (Attempt {attempt+1}): {exc}")
                    break
                except Exception as exc:
                    logger.error(f"Unexpected error in OpenRouter client for model {model} (Attempt {attempt+1}): {exc}")
                    return None, f"Непредвиденная ошибка клиента ИИ: {exc}"

        return None, last_error_msg or "Не удалось получить ответ от OpenRouter после повторных попыток."

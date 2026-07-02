import abc
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from hse_text_analyzer.configuration.settings import AppSettings
from hse_text_analyzer.infrastructure.ai.schemas import (
    AISyntaxResponse,
    AIReferentialResponse,
    AICoordinationalResponse,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


class BaseAIClient(abc.ABC):
    """
    Base class for OpenAI-compatible LLM providers.
    Handles prompt loading, formatting, and schema validation.
    """
    
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._syntax_prompt_template = (PROMPTS_DIR / "syntax_prompt.txt").read_text(encoding="utf-8")
        self._referential_prompt_template = (PROMPTS_DIR / "referential_prompt.txt").read_text(encoding="utf-8")
        self._coordinational_prompt_template = (PROMPTS_DIR / "coordinational_prompt.txt").read_text(encoding="utf-8")

    @abc.abstractmethod
    def _post_with_retry(self, prompt: str) -> tuple[str | None, str | None]:
        pass

    @property
    @abc.abstractmethod
    def _api_key(self) -> str | None:
        pass

    @property
    @abc.abstractmethod
    def _provider_name(self) -> str:
        pass

    def analyze_sentence_syntax(self, sentence_text: str, sentence_structure_text: str) -> AISyntaxResponse:
        if not self._api_key:
            logger.warning(f"{self._provider_name} API key is missing. Skipping AI analysis.")
            return AISyntaxResponse(
                is_ambiguous=False,
                findings=[],
                error_message=f"API ключ {self._provider_name} не задан."
            )

        if len(sentence_text) > 1000:
            logger.warning(f"Sentence is too long ({len(sentence_text)} chars). Skipping AI analysis.")
            return AISyntaxResponse(is_ambiguous=False, findings=[])

        prompt = self._syntax_prompt_template.format(
            sentence_text=sentence_text,
            sentence_structure_text=sentence_structure_text
        )

        content, error_msg = self._post_with_retry(prompt)
        if error_msg:
            return AISyntaxResponse(is_ambiguous=False, findings=[], error_message=error_msg)

        try:
            data = json.loads(content)
            result = AISyntaxResponse.model_validate(data)
            for finding in result.findings:
                if finding.confidence < 0.0 or finding.confidence > 1.0:
                    finding.confidence = max(0.0, min(1.0, finding.confidence))
            return result
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(f"Failed to parse AI response: {exc}. Raw content was: {content}")
            return AISyntaxResponse(is_ambiguous=False, findings=[], error_message="Ошибка декодирования ответа ИИ.")
        except ValidationError as exc:
            logger.error(f"Validation error for AI schema: {exc}")
            return AISyntaxResponse(is_ambiguous=False, findings=[], error_message="Получен некорректный формат ответа от ИИ.")

    def analyze_sentence_reference(self, sentence_text: str, sentence_structure_text: str) -> AIReferentialResponse:
        if not self._api_key:
            logger.warning(f"{self._provider_name} API key is missing. Skipping AI analysis.")
            return AIReferentialResponse(
                is_ambiguous=False,
                findings=[],
                error_message=f"API ключ {self._provider_name} не задан."
            )

        if len(sentence_text) > 1000:
            logger.warning(f"Sentence is too long ({len(sentence_text)} chars). Skipping AI analysis.")
            return AIReferentialResponse(is_ambiguous=False, findings=[])

        prompt = self._referential_prompt_template.format(
            sentence_text=sentence_text,
            sentence_structure_text=sentence_structure_text
        )

        content, error_msg = self._post_with_retry(prompt)
        if error_msg:
            return AIReferentialResponse(is_ambiguous=False, findings=[], error_message=error_msg)

        try:
            data = json.loads(content)
            result = AIReferentialResponse.model_validate(data)
            for finding in result.findings:
                if finding.confidence < 0.0 or finding.confidence > 1.0:
                    finding.confidence = max(0.0, min(1.0, finding.confidence))
            return result
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(f"Failed to parse AI response: {exc}. Raw content was: {content}")
            return AIReferentialResponse(is_ambiguous=False, findings=[], error_message="Ошибка декодирования ответа ИИ.")
        except ValidationError as exc:
            logger.error(f"Validation error for AI schema: {exc}")
            return AIReferentialResponse(is_ambiguous=False, findings=[], error_message="Получен некорректный формат ответа от ИИ.")

    def analyze_sentence_coordination(self, sentence_text: str, sentence_structure_text: str) -> AICoordinationalResponse:
        if not self._api_key:
            logger.warning(f"{self._provider_name} API key is missing. Skipping AI analysis.")
            return AICoordinationalResponse(
                is_ambiguous=False,
                findings=[],
                error_message=f"API ключ {self._provider_name} не задан."
            )

        if len(sentence_text) > 1000:
            logger.warning(f"Sentence is too long ({len(sentence_text)} chars). Skipping AI analysis.")
            return AICoordinationalResponse(is_ambiguous=False, findings=[])

        prompt = self._coordinational_prompt_template.format(
            sentence_text=sentence_text,
            sentence_structure_text=sentence_structure_text
        )

        content, error_msg = self._post_with_retry(prompt)
        if error_msg:
            return AICoordinationalResponse(is_ambiguous=False, findings=[], error_message=error_msg)

        try:
            data = json.loads(content)
            result = AICoordinationalResponse.model_validate(data)
            for finding in result.findings:
                if finding.confidence < 0.0 or finding.confidence > 1.0:
                    finding.confidence = max(0.0, min(1.0, finding.confidence))
            return result
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(f"Failed to parse AI response: {exc}. Raw content was: {content}")
            return AICoordinationalResponse(is_ambiguous=False, findings=[], error_message="Ошибка декодирования ответа ИИ.")
        except ValidationError as exc:
            logger.error(f"Validation error for AI schema: {exc}")
            return AICoordinationalResponse(is_ambiguous=False, findings=[], error_message="Получен некорректный формат ответа от ИИ.")

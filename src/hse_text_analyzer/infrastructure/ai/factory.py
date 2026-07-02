import logging
from hse_text_analyzer.configuration.settings import get_settings
from hse_text_analyzer.infrastructure.ai.base_client import BaseAIClient
from hse_text_analyzer.infrastructure.ai.openrouter_client import OpenRouterClient
from hse_text_analyzer.infrastructure.ai.groq_client import GroqClient

logger = logging.getLogger(__name__)

def get_ai_client() -> BaseAIClient:
    """
    Factory function to get the active AI client based on AppSettings.
    """
    settings = get_settings()
    provider = settings.active_ai_provider.lower()
    
    if provider == "groq":
        logger.info("Instantiating GroqClient")
        return GroqClient()
    else:
        logger.info("Instantiating OpenRouterClient")
        return OpenRouterClient()

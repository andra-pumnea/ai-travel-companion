from app.config import LLMConfig
from app.engine.llm_clients.groq_client import GroqClient


def get_llm_client(config: LLMConfig) -> GroqClient:
    """
    Factory function to get the appropriate LLM client based on the configuration.
    :param config: LLMConfig object containing the configuration for the LLM client.
    :return: An instance of the LLM client.
    """
    if config.provider == "groq":
        return GroqClient().init_client()
    else:
        raise ValueError(f"Unsupported LLM client: {config.provider}")

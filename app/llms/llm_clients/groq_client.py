from groq import Groq

from app.llms.llm_clients.base_llm_client import BaseLLMClient
from app.settings import GroqConfig


class GroqClient(BaseLLMClient):
    """
    A client for interacting with the Groq API.
    """

    def __init__(self):
        self.groq_config = GroqConfig()

    def init_client(self):
        self.client = Groq(
            api_key=self.groq_config.groq_api_key,
        )
        return self.client

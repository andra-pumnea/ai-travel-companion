from typing import Any, Type
from pydantic import BaseModel
import instructor

from app.settings import get_settings
from app.llms.llm_clients.groq_client import GroqClient


class LLMClientManager:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """
        Initializes the LLM client based on the provider.
        :return: An instance of the LLM client.
        """
        client_initializers = {"groq": instructor.from_groq(GroqClient().init_client())}

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self,
        response_model: Type[BaseModel],
        messages: list[dict[str, str]],
        tools: list[dict] = None,
        **kwargs,
    ) -> Any:
        """
        Creates a completion using the LLM client with the provided messages and parameters.
        :param response_model: The Pydantic model to validate the response.
        :param messages: A list of messages to send to the LLM.
        :param kwargs: Additional parameters for the completion request.
        :return: The response from the LLM.
        """
        completion_params = {
            "model": kwargs.get("model", self.settings.model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
            "tools": tools if tools else None,
            "tool_choice": kwargs.get("tool_choice", "auto"),
        }
        return self.client.chat.completions.create(**completion_params)

from abc import ABC, abstractmethod
from typing import Type, Optional

from pydantic import BaseModel


class BaseLLMClient(ABC):
    """
    Base class for LLM clients.
    """

    @abstractmethod
    def generate(
        self,
        response_model: Type[BaseModel],
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        **kwargs,
    ):
        """
        Generates a response from the LLM based on the user query and prompt.

        :param user_query: The query from the user.
        :param prompt: The prompt to be used for the LLM.
        :param response_model: The Pydantic model to validate the response.
        :param kwargs: Additional parameters for the LLM call.
        :return: The response from the LLM.
        """
        pass

    @property
    @abstractmethod
    def client(self):
        """
        Returns the LLM client instance.
        """
        pass

    @property
    @abstractmethod
    def settings(self):
        """
        Returns the settings for the LLM client.
        """
        pass

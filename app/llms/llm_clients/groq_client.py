import logging
from typing import Any, Type, Optional
from httpx import TimeoutException

import instructor
from instructor.exceptions import InstructorRetryException
from pydantic import BaseModel
from groq import Groq

from app.llms.llm_clients.llm_client_base import BaseLLMClient
from app.core.settings import GroqConfig
from app.core.exceptions.llm_exceptions import (
    LLMTimeoutError,
    LLMRateLimitError,
    LLMRequestTooLargeError,
    LLMServiceUnavailableError,
    LLMGenerationError,
    LLMUnexpectedError,
)


class GroqClient(BaseLLMClient):
    """
    A client for interacting with the Groq API.
    """

    def __init__(self):
        # TODO: Make singleton
        self._settings = GroqConfig()
        self._client = instructor.from_groq(Groq(api_key=self._settings.groq_api_key))

    def generate(
        self,
        response_model: Type[BaseModel],
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
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
            "model": kwargs["model"],
            "temperature": kwargs.get("temperature", self._settings.temperature),
            "max_retries": kwargs.get("max_retries", self._settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self._settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": kwargs.get("tool_choice", "auto"),
        }
        try:
            return self.client.chat.completions.create(**completion_params)
        except TimeoutException as e:
            logging.error(f"Groq timeout: {e}")
            raise LLMTimeoutError
        except InstructorRetryException as e:
            status = e.args[0].response.status_code
            logging.error(f"Groq HTTP error {status}: {e.args[0].response.text}")

            if status == 429:
                raise LLMRateLimitError()
            elif status == 413:
                raise LLMRequestTooLargeError()
            elif status >= 500:
                raise LLMServiceUnavailableError()
            elif status >= 400:
                raise LLMGenerationError(f"LLM generation error: {str(e)}")

        except Exception as e:
            logging.error(f"Unhandled LLM error: {str(e)}")
            raise LLMUnexpectedError(str(e))

    @property
    def client(self) -> Groq:
        """
        Returns the Groq client instance.
        :return: The Groq client.
        """
        return self._client

    @property
    def settings(self) -> GroqConfig:
        """
        Returns the Groq configuration settings.
        :return: The GroqConfig instance.
        """
        return self._settings

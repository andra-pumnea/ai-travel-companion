import time
import logging

from typing import Type
from pydantic import BaseModel

from app.llms.llm_clients.llm_router import LLMRouter
from app.exceptions import (
    LLMManagerError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMGenerationError,
)


class LLMManager:
    def __init__(self):
        self.llm, self.settings = LLMRouter.get_client("groq")

    def generate_response(
        self,
        model: str,
        user_query: str,
        prompt: str,
        response_model: Type[BaseModel],
        tools: list[dict] = None,
        conversation_id: str = None,
        max_tokens: int = 400,
    ) -> str:
        """
        Calls the LLM with the provided user query and prompt.
        :param user_query: The query from the user.
        :param prompt: The prompt to be used for the LLM.
        :param max_tokens: The maximum number of tokens to generate in the response.
        :param chat_history: The chat history to maintain context.
        :param response_model: The Pydantic model to validate the response.
        :return: The response from the LLM.
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ]
        # TODO: Add conversation history if available
        return self.llm.generate(
            response_model=response_model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            model=model,
        )

    def call_llm_with_retry(
        self, max_retries: int = 5, retry_backoff_base: int = 2, **kwargs
    ):
        """
        Calls the LLM with retry logic.
        :param max_retries: The maximum number of retries for the LLM call.
        :param kwargs: Additional parameters for the LLM call.
        :return: The response from the LLM.
        """

        for model in self.settings.model:
            logging.info(f"Attempting to call model: {model}")
            for attempt in range(max_retries):
                try:
                    return self.generate_response(model, **kwargs)
                except LLMRateLimitError:
                    wait_time = retry_backoff_base * (2**attempt)
                    logging.info(
                        f"Rate limited. Retrying in {wait_time}s... (attempt {attempt + 1})"
                    )
                    time.sleep(wait_time)
                except LLMServiceUnavailableError as e:
                    logging.error(f"Groq internal error {str(e)}. Retrying...")
                    time.sleep(retry_backoff_base * (2**attempt))
                except LLMGenerationError as e:
                    logging.error(
                        f"Unrecoverable LLM error for model {model}: {str(e)}"
                    )
                    break
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    break
            logging.info(f"Model {model} exhausted retries. Trying next...")
        raise LLMManagerError("All fallback models failed.")

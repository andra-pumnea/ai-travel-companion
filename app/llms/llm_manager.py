import time
import random
import logging

from typing import Type
from pydantic import BaseModel

from app.llms.llm_clients.llm_router import LLMRouter
from app.core.exceptions.llm_exceptions import (
    LLMManagerError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMRequestTooLargeError,
    LLMServiceUnavailableError,
    LLMGenerationError,
    LLMUnexpectedError,
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
        self,
        max_retries: int = 5,
        retry_backoff_base: int = 2,
        jitter_factor: float = 0.5,
        **kwargs,
    ):
        """
        Calls the LLM with retry logic.
        :param max_retries: The maximum number of retries for the LLM call.
        :param kwargs: Additional parameters for the LLM call.
        :return: The response from the LLM.
        """

        for model in self.settings.model:
            logging.info(f"Attempting to call model: {model}")
            start_time = time.time()
            for attempt in range(max_retries):
                try:
                    result = self.generate_response(model, **kwargs)
                    total_elapsed = time.time() - start_time
                    logging.info(
                        f"Model {model} succeeded in {total_elapsed:.2f}s (attempt {attempt + 1})"
                    )
                    return result
                except LLMRateLimitError:
                    wait_time = LLMManager.calculate_backoff_time(
                        retry_backoff_base, attempt, jitter_factor
                    )
                    logging.info(
                        f"Rate limited. Retrying in {wait_time}s... (attempt {attempt + 1})"
                    )
                    time.sleep(wait_time)
                except LLMTimeoutError:
                    wait_time = LLMManager.calculate_backoff_time(
                        retry_backoff_base, attempt, jitter_factor
                    )
                    logging.info(
                        f"Timeout occurred, retrying after {wait_time}s ... (attempt {attempt + 1})"
                    )
                    time.sleep(wait_time)
                except LLMServiceUnavailableError as e:
                    wait_time = LLMManager.calculate_backoff_time(
                        retry_backoff_base, attempt, jitter_factor
                    )
                    logging.error(
                        f"Groq internal error {str(e)}. Retrying in {wait_time}s... (attempt {attempt + 1})"
                    )
                    time.sleep(wait_time)
                except (
                    LLMRequestTooLargeError,
                    LLMGenerationError,
                    LLMUnexpectedError,
                ) as e:
                    logging.error(
                        f"Unrecoverable LLM error for model {model}: {str(e)}"
                    )
                    break
            total_elapsed = time.time() - start_time
            logging.info(
                f"Model {model} exhausted retries in {total_elapsed}s. Trying next..."
            )
        raise LLMManagerError("All fallback models failed.")

    @staticmethod
    def calculate_backoff_time(
        base: int, attempt: int, jitter_factor: float = 0.5
    ) -> float:
        """
        Calculate the backoff time with jitter.
        :param base: The base backoff time.
        :param attempt: The current retry attempt.
        :param jitter_factor: The factor to apply for jitter.
        :return: The calculated backoff time.
        """
        base_wait = base * (2**attempt)
        jitter = random.uniform(0, jitter_factor * base_wait)
        return base_wait + jitter

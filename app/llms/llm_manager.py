import time
import logging

from typing import Type
from pydantic import BaseModel

from app.rag_engine.memory.local_memory import LocalMemory
from app.llms.llm_clients.llm_client_manager import LLMClientManager


class LLMManager:
    def __init__(self):
        self.llm = LLMClientManager(provider="groq")
        self.memory = LocalMemory()

    def generate_response(
        self,
        model: str,
        user_query: str,
        prompt: str,
        conversation_id: str,
        response_model: Type[BaseModel],
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
        return self.llm.create_completion(
            response_model=response_model,
            messages=messages,
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
        for model in self.llm.settings.model:
            logging.info(f"Attempting to call model: {model}")
            for attempt in range(max_retries):
                try:
                    return LLMManager().generate_response(model, **kwargs)
                except Exception as e:
                    if LLMManager._is_rate_limit_error(e):
                        wait_time = retry_backoff_base * attempt
                        logging.warning(
                            f"Rate limited. Retrying in {wait_time}s... (Attempt {attempt})"
                        )
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Error with model {model}: {e}")
                        break
            logging.info(f"Exhausted retries for:{model}, trying next model...")
        raise RuntimeError("All fallback models failed.")

    @staticmethod
    def _is_rate_limit_error(error):
        """
        Checks if the error is a rate limit error.
        :param error: The error to check.
        :return: True if the error is a rate limit error, False otherwise.
        """
        return "429" in str(error).lower()

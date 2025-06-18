from app.engine.llm_clients.llm_client_factory import LLMClientFactory
from typing import Type
from pydantic import BaseModel


class LLMManager:
    def __init__(self):
        self.llm = LLMClientFactory(provider="groq")

    def generate_response(
        self,
        user_query: str,
        prompt,
        chat_history: list,
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
        messages.extend(chat_history[:3])
        return self.llm.create_completion(
            response_model=response_model, messages=messages, max_tokens=max_tokens
        )

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
        )

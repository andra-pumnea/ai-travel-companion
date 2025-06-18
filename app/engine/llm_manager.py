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
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_country_code",
                    "description": "Extracts the 2-letter ISO 3166-1 alpha-2 country code from a user's query, if a country is mentioned.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "country_code": {
                                "type": "string",
                                "description": "The 2-letter ISO country code (e.g., 'JP' for Japan, 'US' for United States). If no country is mentioned, return null",
                            }
                        },
                        "required": ["country_code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "rewrite_query",
                    "description": "Rewrite a follow-up question into a standalone query using conversation context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rewritten_user_query": {
                                "type": "string",
                                "description": "The rewritten standalone version of the user's follow-up query.",
                            },
                        },
                        "required": ["rewritten_user_query"],
                    },
                },
            },
        ]
        return self.llm.create_completion(
            response_model=response_model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

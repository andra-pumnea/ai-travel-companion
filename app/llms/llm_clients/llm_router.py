from typing import Any

from app.llms.llm_clients.groq_client import GroqClient


class LLMRouter:
    _client_registry = {}
    _clients = {"groq": (GroqClient(), GroqClient().settings)}

    @classmethod
    def register_client(cls, provider: str, initializer: tuple[Any, Any]):
        """
        Registers a new LLM client with the router.
        :param provider: The name of the LLM provider.
        :param initializer: A tuple containing the client instance and its settings."""
        cls._client_registry[provider] = initializer

    @classmethod
    def get_client(cls, client: str) -> tuple[Any, Any]:
        """
        Retrieves the LLM client and its settings. Creates and retrieves the client if it does not exist.
        :param client: The name of the LLM client to retrieve.
        :return: A tuple containing the client instance and its settings."""
        if client not in cls._client_registry:
            try:
                cls.register_client(client, cls._clients[client])
            except KeyError:
                raise ValueError(
                    f"Failed to register client {client}. Available clients: {list(cls._clients.keys())}"
                )
        return cls._client_registry[client]  # Returns (client, settings)

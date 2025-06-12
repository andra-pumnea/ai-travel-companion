from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """
    Base class for LLM clients.
    """

    def __init__(self, client_name: str):
        self.client_name = client_name

    def get_client_name(self) -> str:
        """
        Returns the name of the LLM client.
        """
        return self.client_name

    @abstractmethod
    def init_client(self):
        """
        raise NotImplementedError("Subclasses must implement the 'init_client' method.")

        This is an abstract method and must be overridden by subclasses to implement specific client initialization logic.
        """
        pass

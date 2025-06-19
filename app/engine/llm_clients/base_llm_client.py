from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """
    Base class for LLM clients.
    """

    @abstractmethod
    def init_client(self):
        """
        raise NotImplementedError("Subclasses must implement the 'init_client' method.")

        This is an abstract method and must be overridden by subclasses to implement specific client initialization logic.
        """
        pass

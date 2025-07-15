from abc import ABC, abstractmethod


class BaseMemoryStore(ABC):
    """
    Base class for memory management.
    """

    @abstractmethod
    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """
        Adds data to the memory.
        """
        pass

    @abstractmethod
    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """
        Retrieves data from the memory.
        """
        pass

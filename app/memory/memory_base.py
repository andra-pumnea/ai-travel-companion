from abc import ABC, abstractmethod


class BaseMemoryStore(ABC):
    """
    Base class for memory management.
    """

    @abstractmethod
    def add_data(self, key: str, value: any):
        """
        Adds data to the memory.
        """
        pass

    @abstractmethod
    def get_data(self, key: str) -> any:
        """
        Retrieves data from the memory.
        """
        pass

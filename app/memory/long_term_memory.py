from app.memory.memory_base import BaseMemory


class LongTermMemory(BaseMemory):
    """
    Class for managing long-term memory storage.
    """

    def __init__(self):
        self.memory_storage = {}

    def add_data(self, key: str, value: any):
        """
        Adds data to the long-term memory.
        :param key: The key for the data.
        :param value: The value to store.
        """
        self.memory_storage[key] = value

    def get_data(self, key: str) -> any:
        """
        Retrieves data from the long-term memory.
        :param key: The key for the data.
        :return: The stored value or None if not found.
        """
        return self.memory_storage.get(key)

    def clear_memory(self):
        """
        Clears the long-term memory storage.
        """
        self.memory_storage.clear()

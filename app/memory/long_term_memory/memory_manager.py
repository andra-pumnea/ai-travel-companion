from app.memory.memory_store_base import BaseMemoryStore


class MemoryManager(BaseMemoryStore):
    """
    MemoryManager is responsible for managing long-term memory storage.
    It implements the BaseMemoryStore interface.
    """

    def __init__(self, storage_client, embeddings):
        self.storage_client = storage_client
        self.embeddings = embeddings

    def add_data(self, key: str, value: any):
        """
        Adds data to the memory.
        """
        self.storage_client.add(key, value)

    def get_data(self, key: str) -> any:
        """
        Retrieves data from the memory.
        """
        return self.storage_client.get(key)

    def clear_memory(self):
        """
        Clears the memory.
        """
        self.storage_client.clear()

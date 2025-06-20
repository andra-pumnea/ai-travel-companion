from app.rag_engine.memory.memory_base import BaseMemory


class LocalMemory(BaseMemory):
    """
    In-memory storage for managing memory data.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LocalMemory, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.storage = {}

        self.__class__._initialized = True  # Mark as initialized

    def add_data(self, key: str, value: str):
        """
        Adds data to the in-memory storage.
        """
        if key in self.storage:
            self.storage[key].append(value)
        else:
            self.storage[key] = [value]

    def get_data(self, key: str) -> str:
        """
        Retrieves data from the in-memory storage.
        """
        return self.storage.get(key)

    def clear_memory(self):
        """
        Clears the in-memory storage.
        """
        self.storage.clear()

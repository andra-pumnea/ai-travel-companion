import logging

from app.memory.memory_base import BaseMemoryStore


class LocalMemory(BaseMemoryStore):
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
        if not self._initialized:
            self.storage = {}
            self._initialized = True 
            logging.info("LocalMemory conversation history initialized.")

    def add_message(self, conversation_id: str, role: str, content: str):
        """
        Adds a message to the in-memory chat history.
        """
        message = {"role": role, "content": content}
        if conversation_id not in self.storage:
            self.storage[conversation_id] = []
        self.storage[conversation_id].append(message)

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """
        Returns full chat history for the given conversation.
        """
        return self.storage.get(conversation_id, [])

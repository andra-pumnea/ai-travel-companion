import logging
from typing import Any

from app.memory.memory_base import BaseMemoryStore
from app.memory.conversation_history.data_models import (
    ConversationSession,
    SessionState,
    ChatMessage,
)


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
            self.storage: dict[str, ConversationSession] = {}
            self._initialized = True
            logging.info("LocalMemory conversation history initialized.")

    def _get_session(self, conversation_id: str) -> ConversationSession:
        if conversation_id not in self.storage:
            self.storage[conversation_id] = ConversationSession()
        return self.storage[conversation_id]

    def _save_session(self, conversation_id: str, session: ConversationSession) -> None:
        self.storage[conversation_id] = session

    def add_message(self, conversation_id: str, role: str, content: str):
        """
        Adds a message to the in-memory chat history.
        """
        session = self._get_session(conversation_id)
        session.messages.append(ChatMessage(role=role, content=content))

    def get_history(self, conversation_id: str) -> list[ChatMessage]:
        """
        Returns full chat history for the given conversation.
        """
        return self.storage.get(conversation_id, ConversationSession()).messages

    def get_session_state(self, conversation_id: str) -> SessionState:
        """
        Retrieves the current session state for a conversation.
        """
        return self._get_session(conversation_id).session_state

    def update_session_state(
        self, conversation_id: str, updates: dict[str, Any]
    ) -> None:
        """
        Update any part of the session state.
        """
        session = self._get_session(conversation_id)
        for field, value in updates.items():
            if hasattr(session.session_state, field):
                setattr(session.session_state, field, value)
            else:
                logging.error(f"Invalid field '{field}' in SessionState")
        self._save_session(conversation_id, session)

from abc import ABC, abstractmethod
from typing import Any

from app.memory.conversation_history.data_models import (
    SessionState,
    ChatMessage,
)


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
    def get_history(self, conversation_id: str) -> list[ChatMessage]:
        """
        Retrieves data from the memory.
        """
        pass

    @abstractmethod
    def get_session_state(self, conversation_id: str) -> SessionState:
        """
        Returns the current session state for the conversation.
        """
        pass

    @abstractmethod
    def update_session_state(
        self, conversation_id: str, updates: dict[str, Any]
    ) -> None:
        """
        Adds new structured facts to the session state (deduplicated).
        """
        pass

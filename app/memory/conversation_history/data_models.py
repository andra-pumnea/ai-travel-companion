from pydantic import BaseModel, Field
from typing import Optional


class ChatMessage(BaseModel):
    role: str
    content: str


class SessionState(BaseModel):
    user_query: Optional[str] = None
    collected_facts: list[str] = Field(default_factory=list)
    travel_plan: Optional[str] = None


class ConversationSession(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    session_state: SessionState = Field(default_factory=SessionState)

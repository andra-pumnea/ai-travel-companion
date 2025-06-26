import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3
    use_fallback: bool = False


class GroqConfig(LLMSettings):
    groq_api_key: str = Field(os.getenv("GROQ_API_KEY"))
    model: list[str] = Field(
        default_factory=lambda: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    )
    embedding_model: str = Field("sentence-transformers/all-mpnet-base-v2")

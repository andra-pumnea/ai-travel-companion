import os
import logging
from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseSettings):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class GroqConfig(LLMSettings):
    groq_api_key: str = Field(os.getenv("GROQ_API_KEY"))
    model: str = Field("llama-3.3-70b-versatile")
    embedding_model: str = Field("sentence-transformers/all-mpnet-base-v2")


class Settings(BaseSettings):
    groq: GroqConfig = Field(default_factory=GroqConfig)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings

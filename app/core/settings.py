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


class QdrantConfig(BaseSettings):
    """Settings for Qdrant vector store."""

    qdrant_url: str = Field(os.getenv("QDRANT_URL", "http://localhost:6333"))


class APISettings(BaseSettings):
    """Base settings for API configurations."""

    project_name: str = "Travel Journal RAG Assistant API"
    project_description: str = "API for the Travel Journal RAG Assistant, providing endpoints to search travel journals."
    project_version: str = "1.0.0"


class PostgresConfig(BaseSettings):
    """Settings for PostgreSQL database connection."""

    password: str = Field(os.getenv("POSTGRES_PASSWORD", "password"))
    user_name: str = Field(os.getenv("POSTGRES_USER", "postgres"))
    database: str = Field(os.getenv("POSTGRES_DB", "postgres"))
    host: str = Field(os.getenv("POSTGRES_HOST", "localhost"))
    port: int = Field(os.getenv("POSTGRES_PORT", 5432))
    test: str = Field("test")

    @property
    def db_url(self) -> str:
        return f"postgresql+asyncpg://{self.user_name}:{self.password}@{self.host}:{self.port}/{self.database}"

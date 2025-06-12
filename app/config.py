import os
from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    model: str = Field("llama-3.3-70b-versatile")
    provider: str = Field("groq")


class GroqConfig(BaseSettings):
    groq_api_key: str = Field(os.getenv("GROQ_API_KEY"))

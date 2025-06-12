import os
from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    groq_api_key: str = Field(os.getenv("GROQ_API_KEY", ""))

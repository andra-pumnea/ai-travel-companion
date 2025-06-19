from typing import Optional
from pydantic import BaseModel, Field


class QAResponse(BaseModel):
    thought_process: list[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class CountryExtractionResponse(BaseModel):
    country_code: Optional[str] = Field(
        description="The 2-letter ISO country code (e.g., 'JP' for Japan, 'US' for United States). If no country is mentioned, return null"
    )
    thought_process: list[str] = Field(
        description="List of thoughts that the AI assistant had while extracting the country code"
    )


class QueryRewritingResponse(BaseModel):
    thought_process: list[str] = Field(
        description="List of thoughts that the AI assistant had while rewriting the query"
    )
    rewritten_user_query: str = Field(
        description="The rewritten standalone version of the user's follow-up query."
    )

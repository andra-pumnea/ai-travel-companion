from app.prompts.prompt_base import PromptBase
from typing import Type
from pydantic import BaseModel, Field

from app.data.dtos.fact import FactDTO


class FactExtractingResponse(BaseModel):
    """
    Response model for the fact extracting prompt.
    Contains the extracted facts and the thought process of the AI assistant.
    """

    thought_process: list[str] = Field(
        description="List of thoughts that the AI assistant had while extracting facts."
    )
    extracted_facts: list[FactDTO] = Field(
        description="List of facts extracted from the user's query."
    )


class FactExtracting(PromptBase):
    """
    Class to handle fact extracting prompts.
    """

    prompt_name = "fact_extracting"

    @classmethod
    def format(cls, **kwargs) -> str:
        """
        Formats the prompt with the given conversation history and user query.
        :param conversation_history: The conversation history to include in the prompt.
        :param user_query: The user's query to extract facts from.
        :return: The formatted prompt string.
        """
        prompt = cls.build_prompt(
            cls.prompt_name,
            user_id=kwargs.get("user_id"),
            journal_entries=kwargs.get("journal_entries"),
            existing_facts=kwargs.get("existing_facts"),
        )
        cls._log_token_usage(prompt=prompt)
        return prompt

    @classmethod
    def response_model(cls) -> Type[BaseModel]:
        """
        Returns the Pydantic model class expected for the LLM response.
        :return: The Pydantic model class for the response.
        """
        return FactExtractingResponse

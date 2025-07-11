from typing import Type
from pydantic import BaseModel, Field

from app.prompts.prompt_base import PromptBase


class QueryRewritingResponse(BaseModel):
    thought_process: list[str] = Field(
        description="List of thoughts that the AI assistant had while rewriting the query"
    )
    rewritten_user_query: str = Field(
        description="The rewritten standalone version of the user's follow-up query."
    )


class QueryRewriting(PromptBase):
    """
    Class to handle query rewriting prompts.
    """

    prompt_name = "query_rewriting"

    @classmethod
    def format(cls, **kwargs) -> str:
        """
        Formats the prompt with the given conversation history and follow-up question.
        :param conversation_history: The conversation history to include in the prompt.
        :param followup_question: The follow-up question to rewrite.
        :return: The formatted prompt string."""
        prompt = cls.build_prompt(
            cls.prompt_name,
            conversation_history=kwargs.get("conversation_history"),
            followup_question=kwargs.get("followup_question"),
        )
        cls._log_token_usage(prompt=prompt)
        return prompt

    @classmethod
    def response_model(cls) -> Type[BaseModel]:
        """
        Returns the Pydantic model class expected for the LLM response.
        :return: The Pydantic model class for the response.
        """
        return QueryRewritingResponse

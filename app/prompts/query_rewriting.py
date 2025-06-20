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
    def format(self, conversation_history: str, followup_question: str) -> str:
        """
        Formats the prompt with the given conversation history and follow-up question.
        :param conversation_history: The conversation history to include in the prompt.
        :param followup_question: The follow-up question to rewrite.
        :return: The formatted prompt string."""
        return self.build_prompt(
            self.prompt_name,
            conversation_history=conversation_history,
            followup_question=followup_question,
        )

    @classmethod
    def response_model(cls) -> Type[BaseModel]:
        """
        Returns the Pydantic model class expected for the LLM response.
        :return: The Pydantic model class for the response.
        """
        return QueryRewritingResponse

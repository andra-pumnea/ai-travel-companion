from typing import Type
from pydantic import BaseModel, Field

from app.prompts.prompt_base import PromptBase


class QAResponse(BaseModel):
    thought_process: list[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class QuestionAnswering(PromptBase):
    """
    Class to handle question answering prompts.
    """

    prompt_name = "question_answering"

    @classmethod
    def format(cls, **kwargs) -> str:
        """
        Formats the prompt with the given context.
        :param context: The context to include in the prompt.
        :return: The formatted prompt string.
        """
        prompt = cls.build_prompt(
            cls.prompt_name,
            context=kwargs.get("context"),
        )
        cls._log_token_usage(prompt=prompt)
        return prompt

    @classmethod
    def response_model(cls) -> Type[BaseModel]:
        """
        Returns the Pydantic model class expected for the LLM response.
        :return: The Pydantic model class for the response.
        """
        return QAResponse

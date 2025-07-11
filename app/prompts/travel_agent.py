from typing import Optional, Type
from datetime import datetime
from pydantic import BaseModel, Field

from app.prompts.prompt_base import PromptBase


class PlanStepResponse(BaseModel):
    thought_process: str = Field(
        description="List of thoughts used for reasoning that the AI travel assistant had while planning the trip."
    )
    tool: Optional[str] = Field(
        None,
        description="The name of the tool to invoke next. Omit if this is the final travel plan.",
    )
    tool_input: Optional[str] = Field(
        None,
        description="The input to provide to the tool specified in `tool`. Omit if this is the final travel_plan.",
    )
    answer: Optional[str] = Field(
        None,
        description="The final answer consisting of a plan to be shared with the user.",
    )


class TravelAgentPrompt(PromptBase):
    """Class to handle travel agent prompts."""

    prompt_name = "travel_agent"

    @classmethod
    def format(cls, **kwargs) -> str:
        """
        Formats the prompt with the given user query and context.
        :param user_query: The user's question or request.
        :param tool_names: A comma-separated string of available tool names.
        :param context: The context to include in the prompt.
        :return: The formatted prompt string.
        """
        date = datetime.now().strftime("%Y-%m-%d")
        prompt = cls.build_prompt(
            cls.prompt_name,
            user_query=kwargs.get("user_query"),
            context=kwargs.get("context"),
            date=date,
            max_steps=kwargs.get("max_steps", 5),
            current_step=kwargs.get("current_step"),
        )

        cls._log_token_usage(prompt=prompt)
        return prompt

    @staticmethod
    def response_model() -> Type[BaseModel]:
        """
        Returns the Pydantic model class expected for the LLM response.
        :return: The Pydantic model class for the response.
        """
        return PlanStepResponse

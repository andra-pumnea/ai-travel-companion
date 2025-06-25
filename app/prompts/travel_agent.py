from typing import Optional, Type
from datetime import datetime
from pydantic import BaseModel, Field

from app.prompts.prompt_base import PromptBase


class PlanStepResponse(BaseModel):
    thought_process: str = Field(
        description="List of thoughts that the AI travel assistant had while planning the trip."
    )
    action: Optional[str] = Field(
        None,
        description="The name of the tool to invoke next. Omit if this is the final answer.",
    )
    action_input: Optional[str] = Field(
        None,
        description="The input to provide to the tool specified in `action`. Omit if this is the final answer.",
    )
    final_answer: Optional[str] = Field(
        None,
        description="The final answer or plan along with a friendly follow-up question to be shared with the user. Present only in the final response.",
    )


class TravelAgentPrompt(PromptBase):
    """Class to handle travel agent prompts."""

    prompt_name = "travel_agent"

    @classmethod
    def format(
        cls,
        user_query: str,
        tool_info: str,
        context: str,
        max_steps: int = 5,
        current_step: int = 4,
    ) -> str:
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
            user_query=user_query,
            tool_info=tool_info,
            context=context,
            date=date,
            max_steps=max_steps,
            current_step=current_step + 1,
        )

        cls._log_token_usage(prompt=prompt)
        return prompt

    def response_model() -> Type[BaseModel]:
        """
        Returns the Pydantic model class expected for the LLM response.
        :return: The Pydantic model class for the response.
        """
        return PlanStepResponse

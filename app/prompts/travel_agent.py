from typing import Optional, Type
from datetime import datetime
from pydantic import BaseModel, Field

from app.prompts.prompt_base import PromptBase


class TravelPlan(BaseModel):
    destination: str = Field(..., description="The destination of the trip.")
    travel_dates: Optional[str] = Field(
        None, description="The dates or period of the trip."
    )
    duration: Optional[str] = Field(None, description="How long the trip will last.")
    weather_notes: Optional[str] = Field(
        None, description="Summary of expected weather."
    )
    user_preferences: Optional[str] = Field(
        None, description="User interests or preferences used in planning."
    )
    activities: list[str] = Field(..., description="Planned activities or attractions.")
    budget_estimate: Optional[str] = Field(
        None, description="Estimated cost or budget summary."
    )
    additional_tips: Optional[str] = Field(
        None, description="Any other tips or travel advice."
    )


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
        description="The input to provide to the tool specified in `tool`. Omit if this is the final travel plan.",
    )
    final: bool = Field(
        False,
        description="Set to True when this is the final step and a complete travel plan is returned.",
    )
    travel_plan: Optional[TravelPlan] = Field(
        None,
        description="Structured plan output when the planning is complete. Required if `final` is True.",
    )
    answer: Optional[str] = Field(
        None,
        description="Optional natural-language summary of the travel plan, to be shown to the user.",
    )


class TravelAgentPrompt(PromptBase):
    """Class to handle travel agent prompts."""

    prompt_name = "travel_agent"

    @classmethod
    def format(cls, **kwargs) -> str:
        """
        Formats the prompt with the given user query and context.
        :return: The formatted prompt string.
        """
        date = datetime.now().strftime("%Y-%m-%d")
        prompt = cls.build_prompt(
            cls.prompt_name,
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

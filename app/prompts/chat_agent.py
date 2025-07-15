from typing import Optional, Type
from pydantic import BaseModel, Field

from app.prompts.prompt_base import PromptBase

class ChatAgentResponse(BaseModel):
    answer: Optional[str] = Field(
        None,
        description="The natural language response or follow up question to be shown to the user."
    )
    collected_facts: Optional[list[str]] = Field(
        None,
        description="Any structured facts the assistant has extracted so far, such as destination, dates, or preferences."
    )
    ready_to_plan: Optional[bool] = Field(
        False,
        description="Indicates whether enough information has been gathered to trigger the planner agent."
    )

class ChatAgentPrompt(PromptBase):
    prompt_name = "chat_agent"

    @classmethod
    def format(cls, **kwargs) -> str:
        """
        Formats the prompt with the given user query and context.
        :return: The formatted prompt string.
        """
        prompt = cls.build_prompt(
            cls.prompt_name
        )

        cls._log_token_usage(prompt=prompt)
        return prompt
    
    @staticmethod
    def response_model() -> Type[BaseModel]:
        """
        Returns the Pydantic model class expected for the LLM response.
        :return: The Pydantic model class for the response.
        """
        return ChatAgentResponse
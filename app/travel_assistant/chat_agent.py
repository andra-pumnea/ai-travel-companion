from app.prompts.chat_agent import ChatAgentResponse, ChatAgentPrompt
from app.llms.llm_manager import LLMManager


class ChatAgent:
    def __init__(self):
        self.llm_manager = LLMManager()

    def run(
        self,
        user_query: str,
        conversation_id: str,
        max_tokens: int = 400,
    ) -> ChatAgentResponse:
        """
        Handles one conversational step with the chat agent.
        :param user_query: The user's current input.
        :param conversation_id: The ID used to fetch conversation context.
        :param response_model: The expected Pydantic model of the response.
        :param prompt_template: The base prompt (with context rules).
        :param max_tokens: Maximum token count for the response.
        :return: Parsed and validated response model.
        """
        rendered_prompt = ChatAgentPrompt.format()

        response = self.llm_manager.call_llm_with_retry(
            user_query=user_query,
            prompt=rendered_prompt,
            response_model=ChatAgentPrompt.response_model(),
            conversation_id=conversation_id,
            max_tokens=max_tokens,
        )
        return response

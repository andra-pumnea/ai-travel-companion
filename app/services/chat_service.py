import logging

from app.travel_assistant.planner_agent import PlannerAgent
from app.travel_assistant.chat_agent import ChatAgent
from app.prompts.chat_agent import ChatAgentResponse
from app.prompts.planner_agent import PlannerAgentResponse
from app.memory.conversation_history.local_memory import LocalMemory


class ChatService:
    def __init__(self):
        self.chat_agent = ChatAgent()
        self.planner_agent = PlannerAgent()
        self.conversation_history = LocalMemory()

    async def reply(
        self, user_query: str, user_id: str, trip_id: str, conversation_id: str
    ) -> str:
        """
        Handles a user query, determines whether to respond or generate a plan.
        :param user_query: The user’s message.
        :param user_id: Unique identifier for the user.
        :param trip_id: Identifier for the trip being planned.
        :param conversation_id: Identifier for the ongoing conversation.
        :return: Assistant response or generated travel plan.
        """
        self._add_to_conversation_history(conversation_id, user_query, role="user")

        response = self.chat_agent.run(user_query, conversation_id)

        if response.ready_to_plan:
            logging.info(
                f"Conversation {conversation_id} has enough information to start planning."
            )
            return await self._handle_planning(
                response, user_id, trip_id, conversation_id
            )
        else:
            logging.info(
                "Conversation {conversation_id does not have enough information to start planning, asking follow up question."
            )
            return self._handle_follow_up(response, conversation_id)

    def _add_to_conversation_history(
        self, conversation_id: str, message: str, role: str
    ):
        """
        Adds a user message to the conversation history.
        :param conversation_id: Conversation ID to track the history.
        :param message: The user’s input message.
        :param role: Role: user or assistant.
        """
        self.conversation_history.add_message(conversation_id, role, message)

    async def _handle_planning(
        self,
        response: PlannerAgentResponse,
        user_id: str,
        trip_id: str,
        conversation_id: str,
    ) -> str:
        """
        Generates a complete travel plan using the PlannerAgent.
        :param response: LLM response containing collected facts.
        :param user_id: The user's ID.
        :param trip_id: The ID of the current trip.
        :param conversation_id: The conversation ID for accessing message history.
        :return: The final travel plan as a string.
        """
        try:
            facts = ", ".join(response.collected_facts or [])
            initial_query = self._get_initial_user_query(conversation_id)
            full_query = f"{initial_query}, {facts}"

            plan = await self.planner_agent.run(
                user_query=full_query, user_id=user_id, trip_id=trip_id
            )

            self._add_to_conversation_history(
                conversation_id, plan.answer, role="assistant"
            )
            return plan.answer

        except Exception as e:
            raise ValueError(f"Error during trip planning: {e}")

    def _handle_follow_up(
        self, response: ChatAgentResponse, conversation_id: str
    ) -> str:
        """
        Sends a follow-up response to the user if planning isn't ready.
        :param response: The assistant's message from the LLM.
        :param conversation_id: The conversation ID for storing the reply.
        :return: Assistant’s message to the user.
        """
        self._add_to_conversation_history(
            conversation_id, response.answer, role="assistant"
        )
        return response.answer

    def _get_initial_user_query(self, conversation_id: str) -> str:
        """
        Retrieves the initial user query from the conversation history.
        :param conversation_id: Conversation ID to pull history from.
        :return: The first message content from the user.
        """
        history = self.conversation_history.get_history(conversation_id)
        if not history:
            logging.info(
                f"No conversation history found for conversation_id: {conversation_id}"
            )
            return ""
        return history[0]["content"]

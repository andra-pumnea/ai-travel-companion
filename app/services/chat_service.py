import logging

from app.travel_assistant.planner_agent import PlannerAgent
from app.travel_assistant.chat_agent import ChatAgent
from app.prompts.chat_agent import ChatAgentResponse
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
        self.conversation_history.add_message(
            conversation_id, role="user", content=user_query
        )
        updates = {"user_query": user_query}
        self.conversation_history.update_session_state(conversation_id, updates)

        response = self.chat_agent.run(user_query, conversation_id)

        collected_facts = self.conversation_history.get_session_state(
            conversation_id
        ).collected_facts
        collected_facts.extend(response.collected_facts)
        updates = {"collected_facts": list(set(collected_facts))}  # type: ignore
        self.conversation_history.update_session_state(conversation_id, updates)

        if response.ready_to_plan:
            logging.info(
                f"Conversation {conversation_id} has enough information to start planning."
            )
            return await self._handle_planning(
                response, user_id, trip_id, conversation_id
            )
        else:
            logging.info(
                f"Conversation {conversation_id} does not have enough information to start planning, asking follow up question."
            )
            return self._handle_follow_up(response, conversation_id)

    async def _handle_planning(
        self,
        response: ChatAgentResponse,
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
            session = self.conversation_history.get_session_state(conversation_id)
            facts = ", ".join(session.collected_facts or [])
            initial_query = self._get_first_user_query(conversation_id)
            full_query = (
                f"Initial user query: {initial_query}, Collected information: {facts}"
            )

            if session.travel_plan:
                full_query += f", Current travel plan: {session.travel_plan}"

            plan = await self.planner_agent.run(
                user_query=full_query, user_id=user_id, trip_id=trip_id
            )

            self.conversation_history.add_message(
                conversation_id, role="assistant", content=plan.answer
            )
            updates = {"travel_plan": plan.answer}
            self.conversation_history.update_session_state(conversation_id, updates)
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
        self.conversation_history.add_message(
            conversation_id, role="assistant", content=response.answer
        )
        return response.answer

    def _get_first_user_query(self, conversation_id: str) -> str:
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
        return history[0].content

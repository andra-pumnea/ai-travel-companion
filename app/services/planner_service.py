from app.travel_assistant.planner_agent import PlannerAgent
from app.prompts.planner_agent import PlannerAgentResponse


class PlannerService:
    def __init__(self):
        self.planner_agent = PlannerAgent()

    async def plan_trip(
        self, user_query: str, user_id: str, trip_id: str, max_steps: int = 3
    ) -> PlannerAgentResponse:
        """
        Plans a trip based on the user's query and trip ID.

        :param user_query: The query string to plan the trip.
        :param user_trip_id: The unique identifier for the user's trip.
        :return: A string containing the planned trip details.
        """
        try:
            response = await self.planner_agent.run(
                user_query=user_query,
                user_id=user_id,
                trip_id=trip_id,
                max_steps=max_steps,
            )
            return response
        except Exception as e:
            raise ValueError(f"Error during trip planning: {e}")

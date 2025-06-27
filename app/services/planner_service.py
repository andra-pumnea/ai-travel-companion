from app.planner_agent import PlannerAgent


class PlannerService:
    def __init__(self):
        self.planner_agent = PlannerAgent()

    async def plan_trip(
        self, user_query: str, user_trip_id: str, max_steps: int = 5
    ) -> str:
        """
        Plans a trip based on the user's query and trip ID.

        :param user_query: The query string to plan the trip.
        :param user_trip_id: The unique identifier for the user's trip.
        :return: A string containing the planned trip details.
        """
        try:
            response = self.planner_agent.run(
                user_query=user_query, user_trip_id=user_trip_id, max_steps=max_steps
            )
            return response
        except Exception as e:
            raise ValueError(f"Error during trip planning: {e}")

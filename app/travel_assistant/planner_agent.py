import logging

from app.llms.llm_manager import LLMManager
from app.travel_assistant.tools.tool_manager import ToolManager
from app.travel_assistant.tools.weather_tool import WeatherTool
from app.travel_assistant.tools.memory_tool import UserFactsTool

# from app.travel_assistant.tools.retrieval_tool import RetrievalTool
from app.prompts.planner_agent import TravelAgentPrompt
from app.prompts.planner_agent import PlannerAgentResponse


class PlannerAgent:
    def __init__(self):
        self.llm_manager = LLMManager()
        self.tools_manager = ToolManager()
        self.tool_info = self._prepare_tools()

    async def run(
        self, user_query: str, user_id: str, trip_id: str, max_steps: int = 3
    ) -> PlannerAgentResponse:
        """
        Plans a trip based on the user's query by reasoning step-by-step using LLM and tools.

        :param user_query: The user's query to plan for.
        :param user_id: The unique identifier for the user.
        :param trip_id: The unique identifier for the user's trip.
        :param max_steps: The maximum number of planning steps to execute.
        :return: A structured response containing the final travel plan and reasoning process.
        """
        logging.info(f"Running planner agent with user query: {user_query}")

        steps_so_far: list[str] = []
        logging.info(f"Available tools: {self.tool_info}")

        for step in range(max_steps):
            context = "\n".join(steps_so_far)

            rendered_prompt = TravelAgentPrompt.format(
                context=context,
                current_step=step + 1,
                max_steps=max_steps,
            )

            response = self.llm_manager.call_llm_with_retry(
                user_query=user_query,
                prompt=rendered_prompt,
                response_model=TravelAgentPrompt.response_model(),
                tools=self.tools_manager.tool_descriptions,
                max_tokens=1000,
            )

            if response.final:
                logging.info(
                    f"Final answer received at step {step + 1}. Thought process: {response.thought_process}"
                )
                return response

            try:
                step_result = await self._handle_tool_response(
                    response, user_query, user_id, trip_id, step, max_steps
                )
                steps_so_far.append(step_result)
            except Exception as e:
                logging.error(f"Error during planning step {step + 1}: {e}")
                steps_so_far.append(f"Step {step + 1}/{max_steps}: Error: {str(e)}")

        return PlannerAgentResponse(
            answer="Sorry, I couldn't generate a complete plan. Please try again.",
            thought_process="The planner agent was unable to complete the task within the maximum steps allowed.",
            final=True,
        )

    async def _handle_tool_response(
        self,
        response: PlannerAgentResponse,
        user_query: str,
        user_id: str,
        trip_id: str,
        step: int,
        max_steps: int,
    ) -> str:
        """
        Processes a planner step by executing the selected tool and formatting the result.

        :param response: The planner agent's response including tool selection and reasoning.
        :param user_query: The original user query used for context or tool input.
        :param user_id: The unique identifier for the user.
        :param trip_id: The unique identifier for the trip.
        :param step: The current step number in the planning loop.
        :param max_steps: The total number of planning steps allowed.
        :return: A formatted string describing the tool usage and thought process.
        """
        tool_name = response.tool
        tool_input = response.tool_input
        thought_process = response.thought_process

        if not tool_name:
            logging.info("No tool selected for this step.")
            return (
                f"Step {step + 1}/{max_steps}: No tool used. Thought: {thought_process}"
            )

        if tool_input is None:
            tool_input = {}
        tool_input.update(
            {"user_query": user_query, "user_id": user_id, "trip_id": trip_id}
        )
        try:
            tool_response = await self.tools_manager.call_tool(tool_name, **tool_input)
        except Exception as e:
            logging.info(
                f"Step {step + 1}/{max_steps}: Error while calling tool {tool_name}: {str(e)}."
            )
            return f"Step {step + 1}/{max_steps}: Error while calling tool {tool_name}: {str(e)}. Thought: {thought_process}"

        return (
            f"Step {step + 1}/{max_steps}: Tool: {tool_name}, "
            f"Thought process: {thought_process}"
            f"Tool Input: {tool_input}, Tool Output: {tool_response}, "
        )

    def _prepare_tools(self) -> list[str]:
        """
        Prepare the tools for the planner agent.
        :return: A list of tool descriptions.
        """
        logging.info("Preparing tools for the planner agent.")
        # self.tools_manager.register_tool("retrieval_tool", RetrievalTool())
        self.tools_manager.register_tool("weather_tool", WeatherTool())
        self.tools_manager.register_tool("user_facts_tool", UserFactsTool())

        all_tools = []
        for tool in self.tools_manager.tools:
            all_tools.append(f"{tool['name']}: {tool['function_definition']}")
        return all_tools

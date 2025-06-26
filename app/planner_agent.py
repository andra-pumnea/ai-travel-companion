import logging

from app.llms.llm_manager import LLMManager
from app.tools.tool_manager import ToolManager
from app.tools.retrieval_tool import RetrievalTool
from app.tools.weather_tool import WeatherTool
from app.tools.memory_tool import MemoryTool
from app.prompts.travel_agent import TravelAgentPrompt


class PlannerAgent:
    def __init__(self):
        self.llm_manager = LLMManager()
        self.tools_manager = ToolManager()
        self.steps_so_far = []

    def run(self, user_query: str, user_trip_id: str, max_steps: int = 5):
        """
        Generate a plan for the given task using the planner.
        :param user_query: The user's query to plan for.
        :param user_trip_id: Unique identifier for the user's trip.
        :param max_steps: Maximum number of planning steps to take.
        :return: A response model containing the travel plan and thought process.
        """
        logging.info(f"Running planner agent with user query: {user_query}")

        tool_info = self._prepare_tools()
        logging.info(f"Available tools: {tool_info}")

        for step in range(max_steps):
            context = "\n".join(self.steps_so_far)

            rendered_prompt = TravelAgentPrompt.format(
                user_query=user_query,
                context=context,
                current_step=step,
            )

            response = self.llm_manager.call_llm_with_retry(
                user_query=user_query,
                prompt=rendered_prompt,
                response_model=TravelAgentPrompt.response_model(),
                tools=self.tools_manager.tool_descriptions,
                max_tokens=1000,
            )

            if response.answer:
                logging.info(
                    f"Final answer received from the planner agent. Thought process: {response.thought_process}"
                )
                return response

            try:
                action = response.tool
                action_input = response.tool_input
                thought_process = response.thought_process
                logging.info(
                    f"Step {step + 1}: Thought: {response.thought_process}, Action: {action}, Action Input: {action_input}"
                )

                if action and action_input:
                    tool_response = self.tools_manager.get_tool(action).run(
                        action_input
                    )
                    self.steps_so_far.append(
                        f"Step {step + 1}/{max_steps}: Tool: {action}, Input: {action_input}, Response: {tool_response}, Thought Process: {thought_process} \n"
                    )
                else:
                    logging.info("No tool or tool_input generated.")
                    continue
            except Exception as e:
                logging.error(f"Error during planning step {step + 1}: {e}")
                self.steps_so_far.append(
                    f"Step {step + 1}/{max_steps}: Error: {str(e)}"
                )

        return TravelAgentPrompt.response_model()(
            answer="Sorry, I couldn't generate a complete plan. Please try again.",
            thought_process="The planner agent was unable to complete the task within the maximum steps allowed.",
        )

    def _prepare_tools(self) -> list[str]:
        """
        Prepare the tools for the planner agent.
        :return: A list of tool descriptions.
        """
        logging.info("Preparing tools for the planner agent.")
        self.tools_manager.register_tool("retrieval_tool", RetrievalTool())
        self.tools_manager.register_tool("weather_tool", WeatherTool())
        self.tools_manager.register_tool("memory_tool", MemoryTool())

        all_tools = []
        for tool in self.tools_manager.tools:
            all_tools.append(f"{tool['name']}: {tool['function_definition']}")
        return all_tools

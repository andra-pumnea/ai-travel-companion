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

    def run(self, user_query: str, max_steps: int = 5):
        """
        Generate a plan for the given task using the planner.
        """
        logging.info(f"Running planner agent with user query: {user_query}")

        for step in range(max_steps):
            tool_info = self._prepare_tools()
            logging.info(f"Available tools: {tool_info}")

            context = "\n".join(self.steps_so_far)
            logging.info(f"Current context: {context}")

            rendered_prompt = TravelAgentPrompt.format(
                user_query=user_query, tool_info=tool_info, context=context
            )

            response = self.llm_manager.call_llm_with_retry(
                user_query=user_query,
                prompt=rendered_prompt,
                response_model=TravelAgentPrompt.response_model(),
                tools=self.tools_manager.tool_descriptions,
            )

            if response.final_answer:
                logging.info(
                    f"Final answer received from the planner agent. Thought process: {response.thought}"
                )
                return response.final_answer

            try:
                action = response.action
                action_input = response.action_input
                logging.info(
                    f"Step {step + 1}: Thought: {response.thought}, Action: {action}, Action Input: {action_input}"
                )

                if action and action_input:
                    tool_response = self.tools_manager.get_tool(action).run(
                        action_input
                    )
                    self.steps_so_far.append(
                        f"Step {step + 1}/{max_steps}: Tool: {action}, Input: {action_input}, Response: {tool_response} \n"
                    )
                else:
                    logging.info("No further actions required. Ending planning.")
                    break
            except Exception as e:
                logging.error(f"Error during planning step {step + 1}: {e}")
                self.steps_so_far.append(
                    f"Step {step + 1}/{max_steps}: Error: {str(e)}"
                )

        return "Sorry, I couldn't complete the plan in time."

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
            all_tools.append(f"{tool['name']}: {tool['description']}")
        return all_tools

from app.tools.tool_base import ToolBase


class WeatherTool(ToolBase):
    """
    A tool for fetching weather information.
    """

    def __init__(
        self,
        name: str = "weather_tool",
        description: str = "Get current weather information",
    ):
        super().__init__(name, description)

    def run(self, location: str) -> str:
        """
        Fetches the current weather for the specified location.
        :param location: The location to get the weather for.
        :return: A string containing the current weather information.
        """
        # Placeholder for actual weather fetching logic
        return f"The current weather in {location} is sunny with a temperature of 25°C."

    @property
    def tool_info(self) -> dict:
        """
        Returns information about the weather tool.
        :return: A dictionary containing the tool's name and description.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for",
                    }
                },
                "required": ["location"],
            },
        }

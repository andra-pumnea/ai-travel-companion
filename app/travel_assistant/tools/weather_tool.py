from app.travel_assistant.tools.tool_base import ToolBase


class WeatherTool(ToolBase):
    """
    A tool for fetching weather information.
    """

    def __init__(
        self,
        name: str = "weather_tool",
        description: str = "Get current weather information for a specified location.",
    ):
        super().__init__(name, description)
        self.required_keys = ["location"]

    def run(self, **kwargs) -> str:
        """
        Fetches the current weather for the specified location.
        :param location: The location to get the weather for.
        :return: A string containing the current weather information.
        """
        inputs = self._validate_required_inputs(kwargs)
        location = inputs["location"]

        if not location:
            raise ValueError("Missing required input: location")
        # TODO: actual weather fetching logic
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

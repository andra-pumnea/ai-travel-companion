from app.planner_engine.tools.tool_base import ToolBase


class MemoryTool(ToolBase):
    """
    A tool for managing in-memory storage.
    """

    def __init__(
        self,
        name: str = "memory_tool",
        description: str = "Retrieve stored user preferences and interests.",
    ):
        super().__init__(name, description)
        self.memory_storage = {}

    def run(self, user_id: str = None) -> str:
        """
        Adds or retrieves data from the in-memory storage.
        :param key: The key for the data.
        :param value: The value to store or retrieve.
        :return: A confirmation message or the stored value.
        """
        return [
            "Likes hiking, diving, snorkeling, and swimming.",
            "Likes to try new foods.",
            "Doesn't drink alcohol.",
        ]

    @property
    def tool_info(self) -> dict:
        """
        Returns information about the memory tool.
        :return: A dictionary containing the tool's name and description.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID to manage memory for.",
                    }
                },
                "required": ["user_id"],
            },
        }

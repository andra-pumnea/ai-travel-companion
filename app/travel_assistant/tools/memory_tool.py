import logging
from typing import Optional

from app.travel_assistant.tools.tool_base import ToolBase
from app.memory.facts.fact_store import FactStore
from app.server.dependencies import get_storage_client
from app.data.storage.relational_store_base import RelationalStoreBase


class UserFactsTool(ToolBase):
    """
    A tool for managing in-memory storage.
    """

    def __init__(
        self,
        name: str = "user_facts_tool",
        description: str = "Retrieve stored facts about user preferences and interests when traveling",
        storage_client: Optional[RelationalStoreBase] = None,
    ):
        super().__init__(name, description)
        if storage_client is None:
            storage_client = get_storage_client()

        self.fact_store = FactStore(storage_client)
        self.required_keys = ["user_id"]

    async def run(self, **kwargs) -> list[dict]:
        """
        Adds or retrieves data from the in-memory storage.
        :param key: The key for the data.
        :param value: The value to store or retrieve.
        :return: A confirmation message or the stored value.
        """
        inputs = self._validate_required_inputs(kwargs)
        user_id = inputs["user_id"]

        user_facts = await self.fact_store.get_data(user_id)
        logging.info(f"Retrieved {len(user_facts)} facts for user {user_id}.")
        return [fact.model_dump() for fact in user_facts]

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

from app.planner_engine.tools.tool_base import ToolBase
from app.server.dependencies import get_retrieval_pipeline


class RetrievalTool(ToolBase):
    """
    A tool for retrieving information from a specified source.
    """

    def __init__(
        self,
        name: str = "retrieval_tool",
        description: str = "Retrieve using semantic similarity content from the user's past travels.",
    ):
        super().__init__(name, description)
        self.retrieval_pipeline = get_retrieval_pipeline()

    def run(self, query: str, user_trip_id: str) -> list[dict]:
        """
        Retrieves information based on the provided query and metadata.
        :param query: The query to search for in the journal.
        :param metadata: Optional metadata to filter the search.
        :return: A dict containing the retrieved information.
        """
        search_results = self.retrieval_pipeline.search_journal_entries(
            user_query=query, user_trip_id=user_trip_id
        )
        return search_results

    @property
    def tool_info(self) -> dict:
        """
        Returns information about the retrieval tool.
        :return: A dictionary containing the tool's name, description, and parameters.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for in the journal.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata to filter the search.",
                        "additionalProperties": True,
                    },
                },
                "required": ["query"],
            },
        }

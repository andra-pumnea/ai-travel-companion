from app.planner_engine.tools.tool_base import ToolBase
from app.rag_engine.retrieval_pipeline import RetrievalPipeline


class RetrievalTool(ToolBase):
    """
    A tool for retrieving information from a specified source.
    """

    def __init__(
        self,
        name: str = "retrieval_tool",
        description: str = "The retrieval tool is a semantic similarity tool to learn only from the user's past experiences, not future plans.",
    ):
        super().__init__(name, description)
        self.retrieval_pipeline = RetrievalPipeline()

    def run(self, query: str, user_trip_id: str) -> dict:
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

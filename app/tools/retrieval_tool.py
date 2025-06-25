from app.tools.tool_base import ToolBase
from app.rag_engine.retrieval_pipeline import RetrievalPipeline


class RetrievalTool(ToolBase):
    """
    A tool for retrieving information from a specified source.
    """

    def __init__(
        self,
        name: str = "retrieval_tool",
        description: str = "Retrieve information from user's past travel journal to see what they did in the past.",
    ):
        super().__init__(name, description)
        self.retrieval_pipeline = RetrievalPipeline()

    def run(self, query: str, metadata: dict = None) -> dict:
        """
        Retrieves information based on the provided query and metadata.
        :param query: The query to search for in the journal.
        :param metadata: Optional metadata to filter the search.
        :return: A dict containing the retrieved information.
        """
        search_results = self.retrieval_pipeline.search_journal_entries(
            user_query=query, metadata=metadata
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

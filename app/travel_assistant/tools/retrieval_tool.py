from app.travel_assistant.tools.tool_base import ToolBase
from app.server.dependencies import (
    get_retrieval_pipeline,
    get_vector_store_client,
    get_embeddings,
    get_vector_store,
)


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
        storage_client = get_vector_store_client()
        embeddings = get_embeddings()
        vector_store = get_vector_store(
            storage_client=storage_client, embeddings=embeddings
        )
        self.retrieval_pipeline = get_retrieval_pipeline(vector_store=vector_store)
        self.required_keys = ["user_query", "user_id", "trip_id"]

    def run(self, **kwargs) -> list[dict]:
        """
        Retrieves information based on the provided query and metadata.
        :param query: The query to search for in the journal.
        :param metadata: Optional metadata to filter the search.
        :return: A dict containing the retrieved information.
        """
        inputs = self._validate_required_inputs(kwargs)

        user_query = inputs["user_query"]
        user_id = inputs["user_id"]
        trip_id = inputs["trip_id"]
        user_trip_id = f"{user_id}_{trip_id}"

        user_trip_id = f"{user_id}_{trip_id}"
        search_results = self.retrieval_pipeline.search_journal_entries(
            user_query=user_query, user_trip_id=user_trip_id
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

import logging

from app.rag_engine.retrieval_pipeline import RetrievalPipeline


class JournalService:
    def __init__(self, retrieval_pipeline: RetrievalPipeline):
        self.retrieval_pipeline = retrieval_pipeline

    async def search_journal(
        self, user_query: str, user_id: str, trip_id: str, limit: int = 5
    ):
        """
        Search the travel journal based on user query, user ID, and trip ID.

        :param user_query: The query string to search in the journal.
        :param user_trip_id: The unique identifier for the user's trip.
        :param limit: The maximum number of results to return.
        :return: A list of documents matching the query.
        """
        user_trip_id = f"{user_id}_{trip_id}"
        documents = self.retrieval_pipeline.search_journal_entries(
            user_query=user_query, user_trip_id=user_trip_id, limit=limit
        )
        if not documents:
            logging.info(
                f"No documents found for user_id={user_id}, trip_id={trip_id}, query='{user_query}'"
            )
        return documents

    async def search_journal_with_generation(
        self, user_query: str, user_id: str, trip_id: str, limit: int = 5
    ) -> tuple[list[dict], str]:
        """
        Search the travel journal using RAG.

        :param user_query: The query string to search in the journal.
        :param user_id: The unique identifier for the user.
        :param trip_id: The unique identifier for the user's trip.
        :param limit: The maximum number of results to return.
        :return: A list of documents matching the query.
        """
        user_trip_id = f"{user_id}_{trip_id}"
        answer, documents = self.retrieval_pipeline.search_with_generation(
            user_query=user_query, user_trip_id=user_trip_id, limit=limit
        )
        if not documents:
            logging.info(f"No documents found for user_id={user_id}, trip_id={trip_id}")
        return answer, documents

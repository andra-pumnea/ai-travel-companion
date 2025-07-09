import logging

from app.memory.facts.fact_manager import FactManager
from app.rag_engine.retrieval_pipeline import RetrievalPipeline
from app.data.dtos.fact import FactDTO


class FactService:
    """
    Service class to handle fact extraction from travel journal entries.
    This class is responsible for extracting facts from the user's travel journal.
    """

    def __init__(self):
        self.fact_manager = FactManager()
        self.retrieval_pipeline = RetrievalPipeline()

    async def extract_facts(
        self, user_id: str, trip_id: str, limit: int = 5
    ) -> list[FactDTO]:
        """
        Extracts facts from the provided travel journal entries.
        :param journal_entries: List of journal entries to extract facts from.
        :return: A list of FactDTOs containing extracted facts and their categories.
        """
        user_trip_id = f"{user_id}_{trip_id}"
        journal_entries = self.retrieval_pipeline.get_all_journal_entries(user_trip_id)
        if not journal_entries:
            logging.info("No journal entries provided for fact extraction.")
            return []

        facts = await self.fact_manager.extract_facts(user_id, journal_entries[:limit])
        logging.info(
            f"Extracted {len(facts.extracted_facts)} facts for user {user_id} and trip {trip_id}."
        )
        return facts

    async def get_all_facts(self, user_id: str) -> list[FactDTO]:
        """
        Retrieves all facts for a user.
        :param user_id: The ID of the user to retrieve facts for.
        :return: A list of FactDTOs containing the user's facts.
        """
        facts = await self.fact_manager.fact_store.get_data(user_id)
        if not facts:
            logging.info(f"No facts found for user {user_id}.")
            return []

        logging.info(f"Retrieved {len(facts)} facts for user {user_id}.")
        return facts

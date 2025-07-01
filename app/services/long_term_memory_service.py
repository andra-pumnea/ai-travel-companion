import logging

from app.memory.long_term_memory.fact_manager import FactManager
from app.rag_engine.retrieval_pipeline import RetrievalPipeline
from app.data.models.fact import FactDTO


class FactService:
    """
    Service class to handle fact extraction from travel journal entries.
    This class is responsible for extracting facts from the user's travel journal.
    """

    def __init__(self):
        self.fact_manager = FactManager()
        self.retrieval_pipeline = RetrievalPipeline()

    async def extract_facts(self, user_id: str, trip_id: str) -> list[FactDTO]:
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

        return self.fact_manager.extract_facts(journal_entries[:10])

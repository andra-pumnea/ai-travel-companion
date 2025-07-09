import logging

from app.memory.facts.fact_manager import FactManager
from app.data.dtos.fact import FactDTO


class FactService:
    """
    Service class to handle fact extraction from travel journal entries.
    This class is responsible for extracting facts from the user's travel journal.
    """

    def __init__(self):
        self.fact_manager = FactManager()

    async def extract_facts(
        self, user_id: str, trip_id: str, limit: int = 5
    ) -> list[FactDTO]:
        """
        Extracts facts from the provided travel journal entries.
        :param user_id: The ID of the user to extract facts for.
        :param trip_id: The ID of the trip to extract facts from.
        :param limit: The maximum number of journal entries to process in a batch.
        :return: A list of FactDTOs containing extracted facts and their categories.
        """
        facts = await self.fact_manager.extract_facts(user_id, trip_id, limit)
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

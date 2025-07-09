import logging

from app.data.dtos.fact import FactDTO
from app.prompts.fact_extracting import FactExtracting
from app.llms.llm_manager import LLMManager
from app.data.storage.postgres_client import PostgresClientWrapper
from app.memory.facts.fact_store import FactStore


class FactManager:
    """
    Class to manage the extraction and update of facts from travel journal entries.
    """

    def __init__(self):
        self.llm_manager = LLMManager()
        self.fact_store = FactStore(storage_client=PostgresClientWrapper())

    async def extract_facts(
        self, user_id: str, journal_entries: list[dict]
    ) -> list[FactDTO]:
        """
        Extracts facts from the provided travel journal entries.
        :return: A list of dictionaries containing extracted facts and their categories.
        """
        # Convert journal entries to a single string
        journal_entries_str = "\n".join(
            [
                entry.get("description") or ""
                for entry in journal_entries
                if "description" in entry
            ]
        )

        # Use the FactExtracting prompt to extract facts
        rendered_prompt = FactExtracting.format(
            user_id=user_id, journal_entries=journal_entries_str
        )

        logging.info(f"Prompt token usage: {len(rendered_prompt)}")

        # Call the LLM (this is a placeholder, replace with actual LLM call)
        response = self.llm_manager.call_llm_with_retry(
            user_query=journal_entries_str,
            prompt=rendered_prompt,
            response_model=FactExtracting.response_model(),
            max_tokens=400,
        )

        await self.fact_store.add_data(response.extracted_facts)

        return response

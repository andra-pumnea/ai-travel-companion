import logging

from app.data.dtos.fact import FactDTO
from app.prompts.fact_extracting import FactExtracting
from app.llms.llm_manager import LLMManager
from app.memory.facts.fact_store import FactStore
from app.rag_engine.retrieval_pipeline import RetrievalPipeline


class FactManager:
    """
    Class to manage the extraction and update of facts from travel journal entries.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, fact_store: FactStore, retrieval_pipeline: RetrievalPipeline):
        if not self._initialized:
            self.fact_store = fact_store
            self.retrieval_pipeline = retrieval_pipeline
            self.llm_manager = LLMManager()
            self._initialized = True

    async def extract_facts(
        self,
        user_id: str,
        trip_id: str,
        limit: int = 5,
    ) -> list[FactDTO]:
        """
        Extracts facts from the provided travel journal entries.
        :param user_id: The ID of the user to extract facts for.
        :param trip_id: The ID of the trip to extract facts from.
        :param limit: The maximum number of journal entries to process in a batch.
        :return: A list of dictionaries containing extracted facts and their categories.
        """
        user_trip_id = f"{user_id}_{trip_id}"
        journal_entries = self.retrieval_pipeline.get_all_journal_entries(user_trip_id)
        if not journal_entries:
            logging.info("No journal entries provided for fact extraction.")
            return []

        existing_facts = []
        existing_facts_str = ""
        for i in range(0, len(journal_entries), limit):
            batch_entries = journal_entries[i : i + limit]
            logging.info(
                f"Processing batch {i // limit + 1} with {len(batch_entries)} entries for fact extraction."
            )

            journal_entries_str = "\n".join(
                [
                    entry.get("description") or ""
                    for entry in batch_entries
                    if "description" in entry
                ]
            )

            rendered_prompt = FactExtracting.format(
                user_id=user_id,
                journal_entries=journal_entries_str,
                existing_facts=existing_facts_str,
            )
            response = self.llm_manager.call_llm_with_retry(
                user_query=journal_entries_str,
                prompt=rendered_prompt,
                response_model=FactExtracting.response_model(),
                max_tokens=400,
            )

            new_facts = [
                {"category": fact.category, "fact_text": fact.fact_text}
                for fact in response.extracted_facts
            ]
            existing_facts.extend(new_facts)

            existing_facts_str = "\n".join(
                f"- {fact['category']}: {fact['fact_text']}" for fact in existing_facts
            )

        await self.fact_store.add_data(response.extracted_facts)

        return response.extracted_facts

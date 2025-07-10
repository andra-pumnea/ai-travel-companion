from app.memory.memory_base import BaseMemoryStore
from app.data.storage.relational_store_base import RelationalStoreBase
from app.data.dtos.fact import FactDTO

TABLE_NAME = "user_facts"


class FactStore(BaseMemoryStore):
    """
    MemoryManager is responsible for managing long-term memory storage.
    It implements the BaseMemoryStore interface.
    """

    def __init__(self, storage_client: RelationalStoreBase):
        self.storage_client = storage_client

    async def add_data(self, facts: list[FactDTO]):
        """
        Adds data to the memory.
        :param facts: List of FactDTOs to add.
        """
        fact_data = [
            {
                "user_id": fact.user_id,
                "fact": fact.fact_text,
                "category": fact.category,
            }
            for fact in facts
        ]
        await self.storage_client.upsert_records(TABLE_NAME, records=fact_data)

    async def get_data(self, user_id: str) -> list[FactDTO]:
        """
        Retrieves data from the memory.
        :param user_id: The ID of the user to retrieve facts for.
        :return: A list of FactDTOs containing the user's facts.
        """
        query_params = {"user_id": user_id}
        results = await self.storage_client.query(
            table_name=TABLE_NAME, query_params=query_params
        )
        facts = [
            FactDTO(
                user_id=result["user_id"],
                fact_text=result["fact"],
                category=result["category"],
            )
            for result in results
        ]
        return facts

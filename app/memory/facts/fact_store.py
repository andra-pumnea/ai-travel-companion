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
        """
        fact_data = [
            {
                "user_id": fact.user_id,
                "fact": fact.fact_text,
                "category": fact.category,
            }
            for fact in facts
        ]
        await self.storage_client.add_records(TABLE_NAME, records=fact_data)

    def get_data(self, key: str) -> any:
        """
        Retrieves data from the memory.
        """
        return self.storage_client.get(key)

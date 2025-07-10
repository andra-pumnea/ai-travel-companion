from fastapi import Depends

from app.memory.facts.fact_manager import FactManager
from app.memory.facts.fact_store import FactStore
from app.services.facts_service import FactService
from app.data.storage.postgres_client import PostgresClientWrapper
from app.core.settings import PostgresConfig


def get_postgres_client() -> PostgresClientWrapper:
    return PostgresClientWrapper(PostgresConfig())


def get_fact_store(
    storage_client: PostgresClientWrapper = Depends(get_postgres_client),
) -> FactStore:
    return FactStore(storage_client=storage_client)


def get_fact_manager(fact_store: FactStore = Depends(get_fact_store)) -> FactManager:
    return FactManager(fact_store=fact_store)


def get_fact_service(
    fact_manager: FactManager = Depends(get_fact_manager),
) -> FactService:
    return FactService(fact_manager=fact_manager)

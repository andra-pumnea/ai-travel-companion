from fastapi import Depends

from app.memory.facts.fact_manager import FactManager
from app.memory.facts.fact_store import FactStore
from app.services.facts_service import FactService
from app.services.journal_service import JournalService
from app.data.storage.relational_store_base import RelationalStoreBase
from app.data.storage.postgres_client import PostgresClientWrapper
from app.data.storage.qdrant_client import QdrantClientWrapper
from app.data.storage.vector_store_base import VectorStoreBase
from app.core.settings import QdrantConfig
from app.core.settings import PostgresConfig
from app.embeddings.embedding_base import EmbeddingBase
from app.embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from app.rag_engine.vector_store import VectorStore
from app.rag_engine.retrieval_pipeline import RetrievalPipeline


def get_embeddings() -> EmbeddingBase:
    return HuggingFaceEmbeddings()


def get_vector_store_client() -> VectorStoreBase:
    return QdrantClientWrapper(QdrantConfig())


def get_vector_store(
    storage_client: VectorStoreBase = Depends(get_vector_store_client),
    embeddings: EmbeddingBase = Depends(get_embeddings),
) -> VectorStore:
    return VectorStore(storage_client, embeddings)


def get_retrieval_pipeline(
    vector_store: VectorStore = Depends(get_vector_store),
) -> RetrievalPipeline:
    return RetrievalPipeline(vector_store)


def get_storage_client() -> RelationalStoreBase:
    return PostgresClientWrapper(PostgresConfig())


def get_fact_store(
    storage_client: RelationalStoreBase = Depends(get_storage_client),
) -> FactStore:
    return FactStore(storage_client)


def get_fact_manager(
    fact_store: FactStore = Depends(get_fact_store),
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> FactManager:
    return FactManager(fact_store, retrieval_pipeline)


def get_fact_service(
    fact_manager: FactManager = Depends(get_fact_manager),
) -> FactService:
    return FactService(fact_manager)


def get_journal_service(
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
) -> JournalService:
    return JournalService(retrieval_pipeline)

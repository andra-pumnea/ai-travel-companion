import logging

from app.rag_engine.vector_store import VectorStore
from app.settings import QdrantConfig
from app.storage_clients.qdrant_client import QdrantClientWrapper
from app.embeddings.huggingface_embeddings import HuggingFaceEmbeddings


class IndexingPipeline:
    def __init__(self):
        self.storage_client = QdrantClientWrapper(QdrantConfig())
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = VectorStore(self.storage_client, self.embeddings)

    def add_trip_to_vector_store(self, data, user_trip_id: str):
        """
        Reads a trip from Polarsteps, converts it to documents, and adds them to the vector store.
        :param data: The trip data to be indexed.
        :param user_trip_id: Unique identifier for the user's trip collection.
        """
        logging.info(f"Read trip data with {len(data.all_steps)} steps.")

        prepared_data = self.vector_store.prepare_data(data.all_steps)
        collection_name = f"{user_trip_id}_trip_collection"

        try:
            self.vector_store.add_documents(
                collection_name=collection_name, documents=prepared_data
            )
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {e}")
            raise e

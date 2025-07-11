import logging

from qdrant_client.models import PointStruct

from app.data.dtos.trip import TripStepDTO
from app.data.storage.vector_store_base import VectorStoreBase
from app.embeddings.embedding_base import EmbeddingBase
from app.core.exceptions.custom_exceptions import VectorStoreError


class VectorStore:
    """Class to manage the vector store."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, client: VectorStoreBase, embeddings: EmbeddingBase):
        if not self._initialized:
            self.client = client
            self.embeddings = embeddings
            self._initialized = True

    def prepare_data(self, documents: list[TripStepDTO]) -> list[PointStruct]:
        """
        Prepares data for vectorization.
        :param documents: List of documents to prepare.
        :return: List of prepared documents with embeddings.
        """
        texts = [
            doc.description if doc.description else doc.display_name
            for doc in documents
        ]

        embeddings = self.embeddings.embed(texts)

        prepared_documents = [
            VectorStoreBase.trip_step_to_document(dto=doc, embedding=embedding)
            for doc, embedding in zip(documents, embeddings)
        ]
        logging.info(f"Prepared {len(prepared_documents)} documents with embeddings.")
        return prepared_documents

    def add_documents(self, collection_name: str, documents: list[str]):
        """
        Adds documents to the vector store.
        :param collection_name: Name of the collection to add documents to.
        :param documents: List of documents to add.
        """
        if not self.client.collection_exists(collection_name):
            embedding_size = self.embeddings.get_embedding_dimension()
            self.client.create_collection(collection_name, embedding_size)

        self.client.add_documents(collection_name, documents)

    def search(self, collection_name: str, query: str, limit: int = 5):
        """
        Searches for documents in the vector store.
        :param collection_name: Name of the collection to search in.
        :param query: Query string to search for.
        :param limit: Maximum number of results to return.
        :return: List of search results.
        """
        try:
            embedding = self.embeddings.embed(query)
        except Exception as e:
            logging.error(f"Error embedding query: {str(e)}")
            raise VectorStoreError(f"{str(e)}")
        return self.client.search(collection_name, embedding, limit)

    def get_all_documents(self, collection_name: str):
        """
        Retrieves all documents from a specified collection in the vector store.
        :param collection_name: Name of the collection to retrieve documents from.
        :return: List of all documents in the collection.
        """
        try:
            return self.client.get_all_documents(collection_name)
        except Exception as e:
            logging.error(
                f"Error retrieving documents from collection '{collection_name}': {str(e)}"
            )
            raise e

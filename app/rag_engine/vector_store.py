import logging

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams


from app.rag_engine.embeddings import Embeddings


class VectorStore:
    """Class to manage the vector store."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return  # Prevent re-initializing

        self.qdrant_client = QdrantClient(":memory:")

        # Initialize the embeddings
        self.embeddings = Embeddings()

        # Initialize vector store as None
        self.qdrant_vector_store = None

        self.__class__._initialized = True  # Mark as initialized

    def get_vector_store(self) -> QdrantVectorStore:
        """Get the Qdrant vector store."""
        if self.qdrant_vector_store is None:
            raise ValueError(
                "Vector store not initialized. Call create_collection first."
            )
        return self.qdrant_vector_store

    def create_collection(self, collection_name: str):
        """
        Create a new collection in the vector store.
        """
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embeddings.embedding_size, distance=Distance.COSINE
                ),
            )
            logging.info(f"Collection '{collection_name}' created successfully.")
            # Initialize the vector store
            self.qdrant_vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                embedding=self.embeddings.get_embedding_model(),
            )
        except Exception as e:
            logging.error(f"Error creating collection '{collection_name}': {e}")

    def add_documents(self, documents: list) -> list:
        """
        Add documents to the vector store.

        """
        try:
            document_ids = self.qdrant_vector_store.add_documents(documents=documents)
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {e}")
            document_ids = []
        return document_ids

    def similarity_search(self, query: str, metadata: dict = None, k: int = 5) -> list:
        """
        Perform a similarity search in the vector store.

        :param query: The query string to search for.
        :param k: The number of nearest neighbors to return.
        :return: A list of documents that are similar to the query.
        """
        metadata_filter = None
        if metadata:
            metadata_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.location.country_code",
                        match=models.MatchValue(value=metadata.get("country_code", "")),
                    )
                ]
            )
        try:
            results = self.qdrant_vector_store.similarity_search(
                query=query, k=k, filter=metadata_filter
            )
            return results
        except Exception as e:
            logging.error(f"Error during similarity searc, returning empty result: {e}")
            return []

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


from app.engine.embeddings import Embeddings


class VectorStore:
    """Class to manage the vector store."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, collection_name: str):
        if self.__class__._initialized:
            return  # Prevent re-initializing

        self.client = QdrantClient(":memory:")

        # Initialize the embeddings
        self.embeddings = Embeddings()

        # Create the collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.embeddings.embedding_size, distance=Distance.COSINE
            ),
        )

        # Initialize the vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings.get_embedding_model(),
        )
        self.__class__._initialized = True  # Mark as initialized

    def add_documents(self, documents: list) -> list:
        """
        Add documents to the vector store.

        """
        try:
            document_ids = self.vector_store.add_documents(documents=documents)
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            document_ids = []
        return document_ids

    def get_vector_store(self) -> QdrantVectorStore:
        """Get the Qdrant vector store."""
        return self.vector_store

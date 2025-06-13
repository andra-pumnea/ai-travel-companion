from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


from app.engine.embeddings import Embeddings


class VectorStore:
    """Class to manage the vector store."""

    def __init__(self, collection_name: str):
        self.client = QdrantClient(":memory:")

        # Initialize the embeddings
        self.embeddings = Embeddings()

        # Create the collection
        self.collection_name = collection_name
        self.create_collection()

        # Initialize the vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings.get_embedding_model(),
        )

    def create_collection(self):
        """Create a collection in the vector store."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embeddings.embedding_size, distance=Distance.COSINE
            ),
        )

    def add_documents(self, documents: list) -> list:
        """
        Add documents to the vector store.

        """
        document_ids = self.vector_store.add_documents(documents=documents)
        return document_ids

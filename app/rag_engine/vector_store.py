import logging

from app.data.storage.storage_base import StorageBase
from app.embeddings.embedding_base import EmbeddingBase
from app.exceptions import VectorStoreError


class VectorStore:
    """Class to manage the vector store."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, client: StorageBase, embeddings: EmbeddingBase):
        if self.__class__._initialized:
            return  # Prevent re-initializing

        self.client = client
        self.embeddings = embeddings

        self.__class__._initialized = True  # Mark as initialized

    def prepare_data(self, documents: list[str]) -> list:
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
            self.client.trip_step_to_point(dto=doc, embedding=embedding)
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

    def add_document(self, collection_name: str, document: str):
        """
        Adds a single document to the vector store.
        :param collection_name: Name of the collection to add the document to.
        :param document: Document to add.
        """
        self.client.add_document(collection_name, [document])

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

from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    """
    Base class for storage clients.
    """

    @abstractmethod
    def add_documents(self, collection_name: str, documents: list):
        """
        Add documents to the storage.
        """
        pass

    @abstractmethod
    def add_document(self, collection_name, document) -> str:
        """
        Add a single document to the storage.

        :param document: The document to add.
        """
        pass

    @abstractmethod
    def search(self, collection_name: str, query_embedding: list, k: int = 5):
        """
        Perform a similarity search in the storage.

        :param query: The query string to search for.
        :param metadata: Optional metadata to filter results.
        :param k: The number of nearest neighbors to return.
        :return: A list of documents that are similar to the query.
        """
        pass

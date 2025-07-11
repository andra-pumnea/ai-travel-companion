from abc import ABC, abstractmethod
from typing import Any


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
    def search(self, collection_name: str, query_embedding: list, k: int = 5):
        """
        Perform a similarity search in the storage.

        :param query: The query string to search for.
        :param metadata: Optional metadata to filter results.
        :param k: The number of nearest neighbors to return.
        :return: A list of documents that are similar to the query.
        """
        pass

    @abstractmethod
    def get_all_documents(self, collection_name: str):
        """
        Retrieve all documents from a specified collection.

        :param collection_name: Name of the collection to retrieve documents from.
        :return: List of all documents in the collection.
        """
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the storage.

        :param collection_name: Name of the collection to check.
        :return: True if the collection exists, False otherwise.
        """
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, embedding_size: int):
        """
        Create a new collection in the storage.

        :param collection_name: Name of the collection to create.
        :param embedding_size: Size of the embeddings used in the collection.
        """
        pass

    @staticmethod
    @abstractmethod
    def trip_step_to_document(dto, embedding: list[float]) -> Any:
        """
        Convert a TripStepDTO to a document format suitable for storage.

        :param dto: TripStepDTO object containing trip step data.
        :param embedding: Embedding vector for the trip step.
        :return: Document in the format required by the storage.
        """
        pass

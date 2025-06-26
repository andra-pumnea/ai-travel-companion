import logging

from settings import QdrantConfig
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from app.storage_clients.storage_base import StorageBase
from app.data_models.trip import TripStepDTO


class QdrantClientWrapper(StorageBase):
    """Wrapper for Qdrant client to manage vector store operations."""

    def __init__(self, config: QdrantConfig):
        self.client = QdrantClient(url=config.qdrant_url)

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the Qdrant vector store.
        :param collection_name: Name of the collection to check.
        :return: True if the collection exists, False otherwise.
        """
        return self.client.collection_exists(collection_name)

    def create_collection(self, collection_name: str, embedding_size: int):
        """
        Create a new collection in the Qdrant vector store.
        :param collection_name: Name of the collection to create.
        :param embedding_size: Size of the embedding vectors.
        :return: Confirmation message.
        """
        if not self.collection_exists(collection_name):
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "description": VectorParams(
                            size=embedding_size, distance=Distance.COSINE
                        )
                    },
                )
            except Exception as e:
                raise ValueError(f"Error creating collection '{collection_name}': {e}")
            logging.info(
                f"Collection '{collection_name}' created successfully with embedding size {embedding_size}."
            )
        else:
            logging.info(f"Collection '{collection_name}' already exists.")

    def add_documents(self, collection_name: str, documents: list[PointStruct]):
        """
        Add documents to a specified collection in the Qdrant vector store.
        :param collection_name: Name of the collection to add documents to.
        :param documents: List of documents to add.
        :param embeddings: List of embeddings corresponding to the documents.
        :return: List of document IDs added.
        """
        try:
            self.client.upload_points(collection_name=collection_name, points=documents)
        except Exception as e:
            raise ValueError(
                f"Error adding documents to collection '{collection_name}': {e}"
            )
        logging.info(
            f"Added {len(documents)} documents to collection '{collection_name}'."
        )

    def add_document(self, collection_name: str, document: PointStruct):
        """
        Add a single document to a specified collection in the Qdrant vector store.
        :param collection_name: Name of the collection to add the document to.
        :param document: Document to add as a PointStruct.
        :return: Confirmation message.
        """
        try:
            self.client.upsert(collection_name=collection_name, points=[document])
            logging.info(f"Document added to collection '{collection_name}'.")
        except Exception as e:
            raise ValueError(f"Error adding document: {e}")

    def search(self, collection_name: str, query_embedding: list, k: int = 5):
        """
        Perform a similarity search in a specified collection.
        :param collection_name: Name of the collection to search in.
        :param query_embedding: Embedding of the query to search for.
        :param k: Number of nearest neighbors to return.
        :return: List of similar documents and their IDs.
        """
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            using="description",
            limit=k,
        )
        if not results:
            logging.warning(
                f"No results found for query in collection '{collection_name}'."
            )
            return []

        results_payload = [point.payload for point in results.points]
        return results_payload

    @staticmethod
    def trip_step_to_point(dto: TripStepDTO, embedding: list[float]) -> PointStruct:
        """Convert a TripStepDTO to a PointStruct for Qdrant storage.
        :param dto: TripStepDTO object containing trip step data.
        :param embedding: Embedding vector for the trip step.
        :return: PointStruct object ready for Qdrant storage.
        """
        return PointStruct(
            id=dto.id,  # or uuid.uuid4().int if you want unique auto IDs
            vector={"description": embedding},
            payload={
                "display_name": dto.display_name,
                "description": dto.description,
                "location_name": dto.location_name,
                "lat": dto.lat,
                "lon": dto.lon,
                "detail": dto.detail,
                "country_code": dto.country_code,
                "weather_condition": dto.weather_condition,
                "weather_temperature": dto.weather_temperature,
            },
        )

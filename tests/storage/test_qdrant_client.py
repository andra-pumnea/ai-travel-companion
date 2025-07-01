import pytest
from unittest.mock import MagicMock

from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from app.data.storage.qdrant_client import QdrantClientWrapper
from app.core.settings import QdrantConfig
from app.core.exceptions.custom_exceptions import (
    QdrantClientError,
    CollectionNotFoundError,
)


class TestQdrantClient:
    @pytest.fixture(autouse=True)
    def setup_qdrant_client(self):
        config = QdrantConfig(qdrant_url="http://localhost:6333")
        self.wrapper = QdrantClientWrapper(config)
        self.wrapper.client = MagicMock()

    def test_collection_exists(self):
        collection_name = "existing_collection"
        self.wrapper.client.collection_exists.return_value = True

        result = self.wrapper.collection_exists(collection_name)

        assert result is True
        self.wrapper.client.collection_exists.assert_called_once_with(collection_name)

    def test_collection_does_not_exist(self):
        collection_name = "nonexistent_collection"
        self.wrapper.client.collection_exists.return_value = False

        result = self.wrapper.collection_exists(collection_name)

        assert result is False
        self.wrapper.client.collection_exists.assert_called_once_with(collection_name)

    def test_create_collection(self):
        collection_name = "new_collection"
        self.wrapper.client.collection_exists.return_value = False

        self.wrapper.create_collection(collection_name, embedding_size=128)

        self.wrapper.client.create_collection.assert_called_once_with(
            collection_name=collection_name,
            vectors_config={
                "description": VectorParams(size=128, distance=Distance.COSINE)
            },
        )

    def test_create_collection_already_exists(self):
        collection_name = "existing_collection"
        self.wrapper.client.collection_exists.return_value = True

        self.wrapper.create_collection(collection_name, embedding_size=128)

        self.wrapper.client.create_collection.assert_not_called()
        self.wrapper.client.collection_exists.assert_called_once_with(collection_name)

    def test_add_documents(self):
        collection_name = "test_collection"
        documents = [
            PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "doc 1"}),
            PointStruct(id=2, vector=[0.3, 0.4], payload={"text": "doc 2"}),
        ]

        self.wrapper.add_documents(collection_name, documents)

        self.wrapper.client.upload_points.assert_called_once_with(
            collection_name=collection_name,
            points=documents,
        )

    def test_add_documents_error(self):
        collection_name = "test_collection"
        documents = [
            PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "doc 1"}),
        ]
        self.wrapper.client.upload_points.side_effect = Exception("Upload error")

        with pytest.raises(QdrantClientError) as e:
            self.wrapper.add_documents(collection_name, documents)

        assert "Error adding documents to collection" in str(e.value)
        self.wrapper.client.upload_points.assert_called_once()

    def test_add_document(self):
        collection_name = "test_collection"
        document = PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "doc 1"})

        self.wrapper.add_document(collection_name, document)

        self.wrapper.client.upsert.assert_called_once_with(
            collection_name=collection_name, points=[document]
        )

    def test_add_document_error(self):
        collection_name = "test_collection"
        document = PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "doc 1"})
        self.wrapper.client.upsert.side_effect = Exception("Upsert error")

        with pytest.raises(QdrantClientError) as e:
            self.wrapper.add_document(collection_name, document)

        assert "Error adding document" in str(e.value)
        self.wrapper.client.upsert.assert_called_once_with(
            collection_name=collection_name, points=[document]
        )

    def test_search_collection_not_found(self):
        collection_name = "unknown_collection"
        self.wrapper.collection_exists = MagicMock(return_value=False)

        with pytest.raises(CollectionNotFoundError) as e:
            self.wrapper.search(collection_name, query_embedding=[0.1, 0.2])

        assert f"Collection '{collection_name}' does not exist." in str(e.value)

    def test_search_qdrant_error(self):
        self.wrapper.collection_exists = MagicMock(return_value=True)
        self.wrapper.client.query_points.side_effect = Exception("internal error")

        with pytest.raises(QdrantClientError) as e:
            self.wrapper.search("test_collection", query_embedding=[0.1, 0.2])

        assert "Qdrant search error" in str(e.value)

    def test_search_no_results(self):
        self.wrapper.collection_exists = MagicMock(return_value=True)
        self.wrapper.client.query_points.return_value = []

        results = self.wrapper.search("test_collection", query_embedding=[0.1, 0.2])
        assert results == []

    def test_search_with_results(self):
        collection_name = "test_collection"
        self.wrapper.collection_exists = MagicMock(return_value=True)

        mock_point1 = MagicMock(payload={"text": "doc 1"})
        mock_point2 = MagicMock(payload={"text": "doc 2"})
        mock_response = MagicMock(points=[mock_point1, mock_point2])
        self.wrapper.client.query_points.return_value = mock_response

        results = self.wrapper.search(collection_name, query_embedding=[0.1, 0.2])

        assert results == [{"text": "doc 1"}, {"text": "doc 2"}]
        self.wrapper.client.query_points.assert_called_once_with(
            collection_name=collection_name,
            query=[0.1, 0.2],
            using="description",
            limit=5,
        )

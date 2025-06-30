import pytest
from unittest.mock import MagicMock

from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from app.data.storage.qdrant_client import QdrantClientWrapper
from app.settings import QdrantConfig
from app.exceptions import QdrantClientError, CollectionNotFoundError


@pytest.fixture
def mock_qdrant_client():
    config = QdrantConfig(qdrant_url="http://localhost:6333")
    wrapper = QdrantClientWrapper(config)
    wrapper.client = MagicMock()
    return wrapper


class TestQdrantClient:
    def test_collection_exists(self, mock_qdrant_client):
        collection_name = "existing_collection"
        mock_qdrant_client.client.collection_exists.return_value = True

        result = mock_qdrant_client.collection_exists(collection_name)

        assert result is True
        mock_qdrant_client.client.collection_exists.assert_called_once_with(
            collection_name
        )

    def test_collection_does_not_exist(self, mock_qdrant_client):
        collection_name = "nonexistent_collection"
        mock_qdrant_client.client.collection_exists.return_value = False

        result = mock_qdrant_client.collection_exists(collection_name)

        assert result is False
        mock_qdrant_client.client.collection_exists.assert_called_once_with(
            collection_name
        )

    def test_create_collection(self, mock_qdrant_client):
        collection_name = "new_collection"
        mock_qdrant_client.client.collection_exists.return_value = False

        mock_qdrant_client.create_collection(collection_name, embedding_size=128)

        mock_qdrant_client.client.create_collection.assert_called_once_with(
            collection_name=collection_name,
            vectors_config={
                "description": VectorParams(size=128, distance=Distance.COSINE)
            },
        )

    def test_create_collection_already_exists(self, mock_qdrant_client):
        collection_name = "existing_collection"
        mock_qdrant_client.client.collection_exists.return_value = True

        mock_qdrant_client.create_collection(collection_name, embedding_size=128)

        mock_qdrant_client.client.create_collection.assert_not_called()
        mock_qdrant_client.client.collection_exists.assert_called_once_with(
            collection_name
        )

    def test_add_documents(self, mock_qdrant_client):
        collection_name = "test_collection"
        documents = [
            PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "doc 1"}),
            PointStruct(id=2, vector=[0.3, 0.4], payload={"text": "doc 2"}),
        ]

        mock_qdrant_client.add_documents(collection_name, documents)

        mock_qdrant_client.client.upload_points.assert_called_once_with(
            collection_name=collection_name,
            points=documents,
        )

    def test_add_documents_error(self, mock_qdrant_client):
        collection_name = "test_collection"
        documents = [
            PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "doc 1"}),
        ]
        mock_qdrant_client.client.upload_points.side_effect = Exception("Upload error")

        with pytest.raises(QdrantClientError) as e:
            mock_qdrant_client.add_documents(collection_name, documents)

        assert "Error adding documents to collection" in str(e.value)
        mock_qdrant_client.client.upload_points.assert_called_once()

    def test_add_document(self, mock_qdrant_client):
        collection_name = "test_collection"
        document = PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "doc 1"})

        mock_qdrant_client.add_document(collection_name, document)

        mock_qdrant_client.client.upsert.assert_called_once_with(
            collection_name=collection_name, points=[document]
        )

    def test_add_document_error(self, mock_qdrant_client):
        collection_name = "test_collection"
        document = PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "doc 1"})
        mock_qdrant_client.client.upsert.side_effect = Exception("Upsert error")

        with pytest.raises(QdrantClientError) as e:
            mock_qdrant_client.add_document(collection_name, document)

        assert "Error adding document to collection" in str(e.value)
        mock_qdrant_client.client.upsert.assert_called_once_with(
            collection_name=collection_name, points=[document]
        )


def test_search_collection_not_found(mock_qdrant_client):
    collection_name = "unknown_collection"
    mock_qdrant_client.collection_exists = MagicMock(return_value=False)

    with pytest.raises(CollectionNotFoundError) as e:
        mock_qdrant_client.search(collection_name, query_embedding=[0.1, 0.2])

    assert f"Collection '{collection_name}' does not exist." in str(e.value)


def test_search_qdrant_error(mock_qdrant_client):
    mock_qdrant_client.collection_exists = MagicMock(return_value=True)
    mock_qdrant_client.client.query_points.side_effect = Exception("internal error")

    with pytest.raises(QdrantClientError) as e:
        mock_qdrant_client.search("test_collection", query_embedding=[0.1, 0.2])

    assert "Qdrant search error" in str(e.value)


def test_search_no_results(mock_qdrant_client):
    mock_qdrant_client.collection_exists = MagicMock(return_value=True)
    mock_qdrant_client.client.query_points.return_value = []

    results = mock_qdrant_client.search("test_collection", query_embedding=[0.1, 0.2])
    assert results == []


def test_search_with_results(mock_qdrant_client):
    collection_name = "test_collection"
    mock_qdrant_client.collection_exists = MagicMock(return_value=True)

    mock_point1 = MagicMock(payload={"text": "doc 1"})
    mock_point2 = MagicMock(payload={"text": "doc 2"})
    mock_response = MagicMock(points=[mock_point1, mock_point2])
    mock_qdrant_client.client.query_points.return_value = mock_response

    results = mock_qdrant_client.search(collection_name, query_embedding=[0.1, 0.2])

    assert results == [{"text": "doc 1"}, {"text": "doc 2"}]
    mock_qdrant_client.client.query_points.assert_called_once_with(
        collection_name=collection_name,
        query=[0.1, 0.2],
        using="description",
        limit=5,
    )

import pytest
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient
from fastapi import status

from main import app
from app.core.exceptions.custom_exceptions import (
    CollectionNotFoundError,
    LLMManagerError,
)


@pytest.mark.asyncio
class TestSearchJournal:
    @pytest.fixture(autouse=True)
    def setup_sample_request(self):
        self.sample_request = {
            "user_query": "trip to Japan",
            "user_id": "user123",
            "trip_id": "trip456",
            "limit": 2,
        }

    @pytest.fixture(autouse=True)
    def setup_sample_response(self):
        self.sample_response = {
            "documents": [
                {
                    "display_name": "Osaka",
                    "description": "beautiful",
                    "lat": 123.1,
                    "lng": 456.2,
                    "detail": "Japan",
                    "country_code": "JP",
                    "weather_condition": "cloudy",
                    "weather_temperature": 16,
                },
                {
                    "display_name": "Kyoto",
                    "description": "beautiful",
                    "lat": 123.1,
                    "lng": 456.2,
                    "detail": "Japan",
                    "country_code": "JP",
                    "weather_condition": "cloudy",
                    "weather_temperature": 15,
                },
            ]
        }

    async def test_search_journal_success(self):
        with patch(
            "app.server.routers.journal.journal_service.search_journal",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = self.sample_response["documents"]

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.post("/journal/search", json=self.sample_request)

            assert response.status_code == status.HTTP_200_OK
            assert response.json() == self.sample_response
            mock_search.assert_awaited_once_with(
                user_query=self.sample_request["user_query"],
                user_id=self.sample_request["user_id"],
                trip_id=self.sample_request["trip_id"],
                limit=self.sample_request["limit"],
            )

    async def test_search_journal_collection_not_found(self):
        with patch(
            "app.server.routers.journal.journal_service.search_journal",
            new_callable=AsyncMock,
        ) as mock_search:
            collection_name = (
                f"{self.sample_request['user_id']}_{self.sample_request['trip_id']}"
            )
            mock_search.side_effect = CollectionNotFoundError(collection_name)

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.post("/journal/search", json=self.sample_request)

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert response.json() == {
                "detail": f"Collection '{collection_name}' does not exist."
            }

    async def test_search_journal_unexpected_error(self):
        with patch(
            "app.server.routers.journal.journal_service.search_journal",
            new_callable=AsyncMock,
        ) as mock_search:
            error_message = "error"
            mock_search.side_effect = Exception(error_message)

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.post("/journal/search", json=self.sample_request)

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert response.json() == {
                "detail": f"An unexpected error occurred: {error_message}"
            }

    async def test_search_journal_with_generation_success(self):
        with patch(
            "app.server.routers.journal.journal_service.search_journal_with_generation",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = (
                "Generated answer",
                self.sample_response["documents"],
            )

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.post(
                    "/journal/search_with_generation", json=self.sample_request
                )

            assert response.status_code == status.HTTP_200_OK
            assert response.json() == {
                "answer": "Generated answer",
                "documents": self.sample_response["documents"],
            }
            mock_search.assert_awaited_once_with(
                user_query="trip to Japan",
                user_id="user123",
                trip_id="trip456",
                limit=2,
            )

    async def test_search_journal_with_generation_collection_not_found(self):
        with patch(
            "app.server.routers.journal.journal_service.search_journal_with_generation",
            new_callable=AsyncMock,
        ) as mock_search:
            collection_name = (
                f"{self.sample_request['user_id']}_{self.sample_request['trip_id']}"
            )
            mock_search.side_effect = CollectionNotFoundError(collection_name)

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.post(
                    "/journal/search_with_generation", json=self.sample_request
                )

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert response.json() == {
                "detail": f"Collection '{collection_name}' does not exist."
            }

    async def test_search_journal_with_generation_llm_manager_error(self):
        with patch(
            "app.server.routers.journal.journal_service.search_journal_with_generation",
            new_callable=AsyncMock,
        ) as mock_search:
            error_message = "LLM Manager error"
            mock_search.side_effect = LLMManagerError(error_message)

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.post(
                    "/journal/search_with_generation", json=self.sample_request
                )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert response.json() == {"detail": f"LLM Manager Error: {error_message}"}

    async def test_search_journal_with_generation_unexpected_error(self):
        with patch(
            "app.server.routers.journal.journal_service.search_journal_with_generation",
            new_callable=AsyncMock,
        ) as mock_search:
            error_message = "Unexpected error"
            mock_search.side_effect = Exception(error_message)

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.post(
                    "/journal/search_with_generation", json=self.sample_request
                )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert response.json() == {
                "detail": f"Internal server error: {error_message}"
            }

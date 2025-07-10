import pytest
from unittest.mock import AsyncMock
from httpx import ASGITransport, AsyncClient
from fastapi import status

from main import app
from app.data.dtos.fact import FactDTO
from app.core.exceptions.llm_exceptions import LLMManagerError
from app.services.facts_service import FactService
from app.server.dependencies import get_fact_service


@pytest.mark.asyncio
class TestExtractFactsEndpoints:
    @pytest.fixture(autouse=True)
    def setup_sample_request(self):
        self.sample_request = {
            "user_id": "user123",
            "trip_id": "trip456",
            "limit": 2,
        }

    @pytest.fixture(autouse=True)
    def setup_sample_response(self):
        self.sample_response = [
            FactDTO(
                user_id="user123",
                fact_text="Likes adventure",
                category="travel_style",
            ),
            FactDTO(
                user_id="user123",
                fact_text="Like street food",
                category="food",
            ),
        ]

    async def test_extract_facts_success(self):
        mock_fact_service = AsyncMock(spec=FactService)
        mock_fact_service.extract_facts.return_value = self.sample_response

        app.dependency_overrides[get_fact_service] = lambda: mock_fact_service

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/user_facts/extract_facts", json=self.sample_request
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "extracted_facts": [fact.model_dump() for fact in self.sample_response]
        }

    async def test_extract_facts_error(self):
        mock_fact_service = AsyncMock(spec=FactService)
        mock_fact_service.extract_facts.side_effect = Exception("Test error")

        app.dependency_overrides[get_fact_service] = lambda: mock_fact_service

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/user_facts/extract_facts", json=self.sample_request
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json() == {
            "detail": "An error occurred while extracting facts: Test error"
        }

    async def test_extract_facts_llm_error(self):
        mock_fact_service = AsyncMock(spec=FactService)
        mock_fact_service.extract_facts.side_effect = LLMManagerError("LLM Error")

        app.dependency_overrides[get_fact_service] = lambda: mock_fact_service

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/user_facts/extract_facts", json=self.sample_request
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json() == {
            "detail": "An error occurred while extracting facts: LLM Manager Error: LLM Error"
        }

    async def test_get_all_facts_success(self):
        mock_fact_service = AsyncMock(spec=FactService)
        mock_fact_service.get_all_facts.return_value = self.sample_response

        app.dependency_overrides[get_fact_service] = lambda: mock_fact_service

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/user_facts/user123")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "facts": [fact.model_dump() for fact in self.sample_response]
        }

    async def test_get_all_facts_error(self):
        mock_fact_service = AsyncMock(spec=FactService)
        mock_fact_service.get_all_facts.side_effect = Exception("Test error")

        app.dependency_overrides[get_fact_service] = lambda: mock_fact_service

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/user_facts/user123")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json() == {
            "detail": "An error occurred while retrieving facts: Test error"
        }

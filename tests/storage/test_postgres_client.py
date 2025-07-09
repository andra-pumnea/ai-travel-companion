import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.data.storage.postgres_client import PostgresClientWrapper
from app.core.settings import PostgresConfig


class MockSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.asyncio
class TestPostgresClientWrapper:
    @pytest.fixture(autouse=True)
    def setup_postgres_client(self):
        config = PostgresConfig()
        self.wrapper = PostgresClientWrapper(config)

        # Create the mock session
        self.mock_session = AsyncMock()

        # Mock session.begin() as async context manager
        mock_transaction_ctx = AsyncMock()
        mock_transaction_ctx.__aenter__.return_value = None
        mock_transaction_ctx.__aexit__.return_value = None
        self.mock_session.begin = MagicMock(return_value=mock_transaction_ctx)

        # Wrap mock_session in async context manager (for self.session())
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = self.mock_session
        mock_session_ctx.__aexit__.return_value = None
        self.wrapper.session = MagicMock(return_value=mock_session_ctx)

    async def test_add_records_success(self):
        table_name = "user_facts"
        records = [{"user_id": 1, "text": "test record"}]

        self.wrapper._table_exists = AsyncMock(return_value=True)
        self.wrapper._TABLE_MAPPING = {"user_facts": MagicMock()}

        await self.wrapper.add_records(table_name, records)

        self.mock_session.add_all.assert_called_once()

    async def test_add_records_table_not_found(self):
        table_name = "non_existent_table"
        records = [{"user_id": 1, "text": "test record"}]

        with pytest.raises(
            ValueError,
            match="No mapped model found for table 'non_existent_table'.",
        ):
            await self.wrapper.add_records(table_name, records)

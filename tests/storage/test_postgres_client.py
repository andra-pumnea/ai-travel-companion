import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import Select

from app.data.storage.postgres_client import PostgresClientWrapper
from app.data.storage.db_models import UserFacts
from app.core.settings import PostgresConfig


@pytest.mark.asyncio
class TestPostgresClientWrapper:
    @pytest.fixture(autouse=True)
    def setup_postgres_client(self):
        config = PostgresConfig()
        self.wrapper = PostgresClientWrapper(config)

        # Create the mock session
        self.mock_session = AsyncMock(spec=AsyncSession)

        # Mock session.begin() as async context manager
        mock_transaction_ctx = AsyncMock()
        mock_transaction_ctx.__aenter__.return_value = None
        mock_transaction_ctx.__aexit__.return_value = None
        self.mock_session.begin = MagicMock(return_value=mock_transaction_ctx)

        # Wrap mock_session in async context manager
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = self.mock_session
        mock_session_ctx.__aexit__.return_value = None
        self.wrapper.session = MagicMock(return_value=mock_session_ctx)

    async def test_get_record_type_table_does_not_exist(self):
        table_name = "non_existent_table"

        with pytest.raises(
            ValueError,
            match="Table 'non_existent_table' does not exist in the mapping.",
        ):
            await self.wrapper._get_record_type(table_name)

    async def test_get_record_type_mapping_missing(self):
        table_name = "unknown_table"
        self.wrapper._table_exists = AsyncMock(return_value=True)
        self.wrapper._TABLE_MAPPING = {}

        with pytest.raises(
            ValueError, match="No mapped model found for table 'unknown_table'."
        ):
            await self.wrapper._get_record_type(table_name)

    async def test_execute_db_operation_success(self):
        table_name = "test_table"
        records = [{"user_id": 1, "text": "test record"}]
        record_type = MagicMock()

        self.wrapper._get_record_type = AsyncMock(return_value=record_type)
        self.wrapper._table_exists = AsyncMock(return_value=True)

        mock_operation = AsyncMock()

        await self.wrapper._execute_db_operation(records, table_name, mock_operation)

        mock_operation.assert_awaited_once_with(self.mock_session, record_type, records)

    async def test_sqlalchemy_error_raises_and_logs(self, caplog):
        table_name = "test_table"
        records = [{"user_id": 1, "text": "test record"}]
        record_type = MagicMock()

        self.wrapper._get_record_type = AsyncMock(return_value=record_type)

        mock_operation = AsyncMock(side_effect=SQLAlchemyError())

        with pytest.raises(SQLAlchemyError):
            await self.wrapper._execute_db_operation(
                records, table_name, mock_operation
            )

        assert (
            f"Database error during operation {mock_operation.__name__} on '{table_name}'"
            in caplog.text
        )

    async def test_generic_exception_raises_and_logs(self, caplog):
        table_name = "test_table"
        records = [{"user_id": 1, "text": "test record"}]
        record_type = MagicMock()

        self.wrapper._get_record_type = AsyncMock(return_value=record_type)

        mock_operation = AsyncMock(side_effect=Exception())

        with pytest.raises(Exception):
            await self.wrapper._execute_db_operation(
                records, table_name, mock_operation
            )

        assert (
            f"Unhandled error during operation {mock_operation.__name__} on '{table_name}'"
            in caplog.text
        )

    async def test_add_records_success(self):
        table_name = "test_table"
        record_type = MagicMock()
        records = [{"user_id": 1, "text": "test record"}]

        self.wrapper._table_exists = AsyncMock(return_value=True)
        self.wrapper._TABLE_MAPPING = {"test_table": record_type}

        await self.wrapper.add_records(table_name, records)

        self.mock_session.add_all.assert_called_once()
        record_type.assert_called_once_with(**records[0])

    async def test_add_invalid_records_exception(self):
        self.wrapper = PostgresClientWrapper(PostgresConfig())
        self.wrapper._TABLE_MAPPING = {"user_facts": UserFacts}

        self.wrapper._table_exists = AsyncMock(return_value=True)

        bad_data = [{"user_id": 123}]

        with pytest.raises(SQLAlchemyError):
            await self.wrapper.add_records("user_facts", bad_data)

    async def test_upsert_records_success(self):
        table_name = "test_table"
        records = [{"user_id": 1, "fact": "test record", "category": "test category"}]

        self.wrapper._table_exists = AsyncMock(return_value=True)
        self.wrapper._TABLE_MAPPING = {"test_table": UserFacts}

        await self.wrapper.upsert_records(table_name, records)

        assert self.mock_session.execute.await_count == len(records)

    async def test_upsert_records_missing_metadata(self):
        table_name = "test_table"
        mock_record_type = MagicMock()
        mock_record_type.__name__ = "MockRecordType"
        mock_record_type.get_upsert_conflict_target.return_value = []
        mock_record_type.get_upsert_update_fields.return_value = []

        self.wrapper._TABLE_MAPPING = {table_name: mock_record_type}
        self.wrapper._table_exists = AsyncMock(return_value=True)

        with pytest.raises(
            ValueError,
            match=f"Model '{mock_record_type.__name__}' is missing upsert metadata.",
        ):
            await self.wrapper.upsert_records(table_name, [{"id": 1}])

    async def test_upsert_records_data_modified(self):
        table_name = "test_table"
        records = [
            {"user_id": 1, "fact": "test record old", "category": "test category"}
        ]

        self.wrapper._table_exists = AsyncMock(return_value=True)
        self.wrapper._TABLE_MAPPING = {table_name: UserFacts}

        with patch("app.data.storage.postgres_client.insert") as mock_insert:
            mock_insert_obj = MagicMock()
            mock_insert.return_value = mock_insert_obj

            mock_insert_obj.values.return_value = mock_insert_obj
            mock_insert_obj.on_conflict_do_update.return_value = mock_insert_obj

            self.mock_session.execute.side_effect = AsyncMock()

            await self.wrapper.upsert_records(table_name, records)

            mock_insert.assert_called_once_with(UserFacts)

            mock_insert_obj.values.assert_called_once_with(**records[0])

            call_args = mock_insert_obj.on_conflict_do_update.call_args
            assert call_args is not None

            kwargs = call_args.kwargs
            assert "index_elements" in kwargs
            assert "set_" in kwargs

            assert kwargs["index_elements"] == ["user_id", "category"]
            assert set(kwargs["set_"].keys()) == {"fact"}

    async def test_query_success(self):
        table_name = "test_table"
        query_params = {"user_id": 1}
        mock_record = UserFacts(user_id=1, fact="test fact", category="test category")

        self.wrapper._table_exists = AsyncMock(return_value=True)
        self.wrapper._TABLE_MAPPING = {table_name: UserFacts}

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_record]
        mock_result.scalars.return_value = mock_scalars

        execute_calls = []

        async def mock_execute(stmt, *args, **kwargs):
            execute_calls.append(stmt)
            return mock_result

        self.mock_session.execute = mock_execute

        results = await self.wrapper.query(table_name, query_params)

        assert len(execute_calls) == 1
        assert isinstance(execute_calls[0], Select)

        assert len(results) == 1
        assert results[0]["user_id"] == 1
        assert results[0]["fact"] == "test fact"
        assert results[0]["category"] == "test category"

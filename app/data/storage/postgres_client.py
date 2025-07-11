import logging
from typing import Any, Callable

from sqlalchemy import select, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert

from app.data.storage.relational_store_base import RelationalStoreBase
from app.core.settings import PostgresConfig
from app.data.storage.db_models import UserFacts


class PostgresClientWrapper(RelationalStoreBase):
    """
    Wrapper for PostgreSQL client to manage relational store operations.
    """

    _TABLE_MAPPING = {
        "user_facts": UserFacts,
    }

    def __init__(self, config: PostgresConfig):
        self._config = config
        self.engine = create_async_engine(self._config.db_url)
        self.session = async_sessionmaker(self.engine, expire_on_commit=False)

    async def _table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        :param table_name: The name of the table to check.
        :return: True if the table exists, False otherwise."""

        async with self.engine.connect() as conn:
            return await conn.run_sync(
                lambda sync_conn: table_name in inspect(sync_conn).get_table_names()
            )

    async def _get_record_type(self, table_name: str) -> Any:
        """Check table exists and get record type or raise.
        :param table_name: The name of the table to check.
        :return: The SQLAlchemy model class for the table.
        """
        table_exists = await self._table_exists(table_name)
        if not table_exists:
            raise ValueError(f"Table '{table_name}' does not exist in the mapping.")

        record_type = self._TABLE_MAPPING.get(table_name)
        if record_type is None:
            raise ValueError(f"No mapped model found for table '{table_name}'.")
        return record_type

    async def _execute_db_operation(
        self, records: list[dict], table_name: str, operation_func: Callable
    ) -> None:
        """
        Helper method to execute DB operations within a transaction.
        :param records: List of record dicts to operate on.
        :param table_name: Name of the target table.
        :param operation_func: Coroutine function accepting (session, record_type, records).
        """
        record_type = await self._get_record_type(table_name)
        try:
            async with self.session() as session:
                async with session.begin():
                    await operation_func(session, record_type, records)
            logging.info(
                f"Operation {operation_func.__name__} completed on {len(records)} records in '{table_name}'."
            )
        except SQLAlchemyError as e:
            logging.error(
                f"Database error during operation {operation_func.__name__} on '{table_name}': {e}"
            )
            raise e
        except TypeError as e:
            logging.error(
                f"Type error during operation {operation_func.__name__} on '{table_name}': {e}"
            )
            raise e
        except Exception as e:
            logging.error(
                f"Unhandled error during operation {operation_func.__name__} on '{table_name}': {e}"
            )
            raise e

    async def add_records(self, table_name: str, records: list[dict]) -> None:
        """
        Add a single record to the storage.
        :param table_name: The name of the table to add the record to.
        :param records: The records to add.
        :return: None
        """

        async def inner_add_operation(session, record_type, records):
            new_records = [record_type(**data) for data in records]
            session.add_all(new_records)

        await self._execute_db_operation(records, table_name, inner_add_operation)

    async def upsert_records(self, table_name: str, records: list[dict]) -> None:
        """
        Upsert records into the storage.
        :param table_name: The name of the table to upsert records into.
        :param records: The records to upsert.
        """

        async def inner_upsert_operation(session, record_type, records):
            conflict_keys = record_type.get_upsert_conflict_target()
            update_fields = record_type.get_upsert_update_fields()
            if not conflict_keys or not update_fields:
                raise ValueError(
                    f"Model '{record_type.__name__}' is missing upsert metadata."
                )

            for record in records:
                insert_query = insert(record_type).values(**record)
                update_dict = {
                    field: getattr(insert_query.excluded, field)
                    for field in update_fields
                }
                stmt = insert_query.on_conflict_do_update(
                    index_elements=conflict_keys,
                    set_=update_dict,
                )
                await session.execute(stmt)

        await self._execute_db_operation(records, table_name, inner_upsert_operation)

    async def query(self, table_name: str, query_params: dict) -> list[dict]:
        """
        Query records from the storage.
        :param table_name: The name of the table to query.
        :param query_params: The parameters for the query.
        :return: A list of records that match the query.
        """
        record_type = await self._get_record_type(table_name)

        try:
            async with self.session() as session:
                query = select(record_type)

                for column, value in query_params.items():
                    if not hasattr(record_type, column):
                        logging.info(f"Ignoring invalid column: {column}")
                        continue
                    query = query.where(getattr(record_type, column) == value)

                results = await session.execute(query)
                records = results.scalars().all()

                logging.info(
                    f"Queried {len(records)} records from table '{table_name}'."
                )
                return [
                    PostgresClientWrapper._model_to_dict(record) for record in records
                ]
        except SQLAlchemyError as e:
            logging.error(f"Error querying table '{table_name}': {e}")
            raise e
        except Exception as e:
            logging.error(f"Unexpected error querying table '{table_name}': {e}")
            raise e

    @staticmethod
    def _model_to_dict(db_model_instance: Any) -> dict:
        """
        Convert a SQLAlchemy model instance to a dictionary.
        :param db_model_instance: The SQLAlchemy model instance to convert.
        :return: A dictionary representation of the model instance.
        """
        if db_model_instance is None:
            return {}
        return {
            column.name: getattr(db_model_instance, column.name)
            for column in db_model_instance.__table__.columns
        }

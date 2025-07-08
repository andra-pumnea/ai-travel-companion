import logging

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.data.storage.relational_store_base import RelationalStoreBase
from app.core.settings import PostgressConfig
from app.data.storage.db_models import UserFacts

TABLE_MAPPING = {
    "user_facts": UserFacts,
}


class PostgresClientWrapper(RelationalStoreBase):
    """
    Wrapper for PostgreSQL client to manage relational store operations.
    """

    def __init__(self, config: PostgressConfig):
        self._config = config
        self.engine = create_async_engine(self._config.db_url)
        self.session = async_sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def _table_exists(self, table_name: str, schema: str = "public") -> bool:
        query = text(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = :schema 
                AND table_name = :table_name
            );
            """
        )
        async with self.engine.connect() as conn:
            result = await conn.execute(
                query, {"schema": schema, "table_name": table_name}
            )
            exists = await result.scalar()
        return exists

    async def add_records(self, table_name: str, records: list[dict]) -> None:
        """
        Add a single record to the storage.
        :param table_name: The name of the table to add the record to.
        :param records: The records to add.
        :return: None
        """

        if not await self._table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist in the mapping.")

        record_type = TABLE_MAPPING.get(table_name)
        if record_type is None:
            raise ValueError(f"No mapped model found for table '{table_name}'.")

        try:
            async with self.session() as session:
                async with session.begin():
                    new_records = [record_type(**data) for data in records]
                    session.add_all(new_records)
                logging.info(f"Added {len(records)} records to table '{table_name}'.")
        except Exception as e:
            logging.error(f"Error adding records to table '{table_name}': {e}")
            raise e

    async def query(self, table_name: str, query_params: dict) -> list[dict]:
        """
        Query records from the storage.
        :param table_name: The name of the table to query.
        :param query_params: The parameters for the query.
        :return: A list of records that match the query.
        """
        if not await self._table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist in the mapping.")

        record_type = TABLE_MAPPING.get(table_name)
        if record_type is None:
            raise ValueError(f"No mapped model found for table '{table_name}'.")

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
                return [record.to_dict() for record in records]
        except SQLAlchemyError as e:
            logging.error(f"Error querying table '{table_name}': {e}")
            raise e
        except Exception as e:
            logging.error(f"Unexpected error querying table '{table_name}': {e}")
            raise e

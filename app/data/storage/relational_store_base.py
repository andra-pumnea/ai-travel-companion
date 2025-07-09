from abc import ABC, abstractmethod


class RelationalStoreBase(ABC):
    """
    Base class for relational storage clients.
    """

    @abstractmethod
    def add_records(self, table_name: str, record: list[dict]) -> str:
        """
        Add a single record to the storage.

        :param table_name: The name of the table to add the record to.
        :param record: The record to add.
        :return: The ID of the added record.
        """
        pass

    @abstractmethod
    def upsert_records(self, table_name: str, records: list[dict]) -> None:
        """
        Upsert records into the storage.

        :param table_name: The name of the table to upsert records into.
        :param records: The records to upsert.
        """
        pass

    @abstractmethod
    def query(self, table_name: str, query_params: dict) -> list:
        """
        Query records from the storage.

        :param table_name: The name of the table to query.
        :param query_params: The parameters for the query.
        :return: A list of records that match the query.
        """
        pass

import logging
from app.data_management.data_loader import (
    read_trip_from_polarsteps,
    convert_trip_to_documents,
)
from app.rag_engine.vector_store import VectorStore


class IndexingPipeline:
    @staticmethod
    def add_trip_to_vector_store():
        """
        Reads a trip from Polarsteps, converts it to documents, and adds them to the vector store.
        """
        # Read the trip data from Polarsteps
        trip = read_trip_from_polarsteps()
        logging.info(f"Read trip data with {len(trip.all_steps)} steps.")

        # Convert the trip data to documents
        documents = convert_trip_to_documents(trip)
        logging.info(f"Converted trip data to {len(documents)} documents.")

        # Add the documents to the vector store and return their IDs
        vector_store = VectorStore()
        try:
            vector_store.client.get_collection("trip_collection")
        except Exception as e:
            logging.warning(
                "Collection 'trip_collection' does not exist. Creating a new collection."
            )
            vector_store.create_collection("trip_collection")
        document_ids = vector_store.add_documents(documents=documents)
        logging.info(f"Added {len(document_ids)} documents to the vector store.")

        return document_ids

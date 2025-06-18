from app.data_management.data_loader import (
    read_trip_from_polarsteps,
    convert_trip_to_documents,
)
from app.engine.vector_store import VectorStore


class IndexingPipeline:
    @staticmethod
    def add_trip_to_vector_store():
        """
        Reads a trip from Polarsteps, converts it to documents, and adds them to the vector store.
        """
        # Read the trip data from Polarsteps
        trip = read_trip_from_polarsteps()
        print(f"Trip loaded with {len(trip.all_steps)} steps.")

        # Convert the trip data to documents
        documents = convert_trip_to_documents(trip)
        print(f"Converted trip to {len(documents)} documents.")

        # Add the documents to the vector store and return their IDs
        vector_store = VectorStore()
        try:
            vector_store.client.get_collection("trip_collection")
        except Exception as e:
            print("Collection not found, creating a new one")
            vector_store.create_collection("trip_collection")
        document_ids = vector_store.add_documents(documents=documents)
        print(f"Added {len(document_ids)} documents to the vector store.")

        return document_ids

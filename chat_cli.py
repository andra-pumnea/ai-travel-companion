import sys
import logging

from app.rag_engine.indexing_pipeline import IndexingPipeline
from app.data.io.data_loader import (
    read_trip_from_polarsteps,
)



def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


if __name__ == "__main__":
    setup_logging()

    print("📒 Travel Journal RAG Assistant (type 'exit' to quit)\n")

    # Read the trip data from Polarsteps
    trip_data = read_trip_from_polarsteps()
    user_trip_id = f"{trip_data.user_id}_{trip_data.id}"

    indexing_pipeline = IndexingPipeline()
    try:
        indexing_pipeline.add_trip_to_vector_store(trip_data, user_trip_id)
    except Exception as e:
        logging.error(f"Error during indexing: {e}")
        print(
            "⚠️ An error occurred while indexing the trip data. Please check the logs."
        )
        sys.exit(1)

    logging.info("Trip data indexed successfully.")
    print("🗂️ Trip data indexed successfully.")

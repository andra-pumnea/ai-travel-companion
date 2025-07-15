import sys
import uuid
import logging
import requests # type: ignore

from app.rag_engine.indexing_pipeline import IndexingPipeline
from app.data.io.data_loader import (
    read_trip_from_polarsteps,
)


BASE_URL = "http://localhost:8000"


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def index_data():
    # Read the trip data from Polarsteps
    trip_data = read_trip_from_polarsteps()
    user_trip_id = f"{trip_data.user_id}_{trip_data.id}"

    indexing_pipeline = IndexingPipeline()
    try:
        indexing_pipeline.add_trip_to_vector_store(trip_data, user_trip_id)
        logging.info("Trip data indexed successfully.")
        print("🗂️ Trip data indexed successfully.")
    except Exception as e:
        logging.error(f"Error during indexing: {e}")
        print(
            "⚠️ An error occurred while indexing the trip data. Please check the logs."
        )
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()

    print("📒 Travel Journal RAG Assistant (type 'exit' to quit)\n")
    # index_data()
    user_id = "13574223"
    trip_id = "16018145"
    conversation_id = str(uuid.uuid4())

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        try:
            payload = {
                "user_query": user_input,
                "user_id": user_id,
                "trip_id": trip_id,
                "conversation_id": conversation_id,
            }
            response = requests.post(f"{BASE_URL}/chat/reply", json=payload)
            if response.status_code == 200:
                data = response.json()
                print("Bot:", data.get("answer", "(no reply)"))
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print("Error connecting to the server:", e)

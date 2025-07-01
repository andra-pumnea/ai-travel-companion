import sys
import uuid
import logging

from app.memory.conversation_history.local_memory import LocalMemory
from app.rag_engine.indexing_pipeline import IndexingPipeline
from app.rag_engine.retrieval_pipeline import RetrievalPipeline
from app.data.io.data_loader import (
    read_trip_from_polarsteps,
)

from app.planner_engine.planner_agent import PlannerAgent


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

    chat_history = LocalMemory()
    planner_agent = PlannerAgent()
    retrieval_pipeline = RetrievalPipeline()
    conversation_id = uuid.uuid4().hex

    try:
        while True:
            question = input("\n❓ Ask a question: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            if question == "":
                continue

            try:
                # response = planner_agent.run(
                #     user_query=question, user_trip_id=user_trip_id
                # )
                response = retrieval_pipeline.run(
                    user_query=question, user_trip_id=user_trip_id
                )
            except Exception as e:
                logging.error(f"Error during retrieval: {e}")
                print(
                    "⚠️ An error occurred while processing your request. Please try again."
                )
                continue

            print(f"💬 Answer: {response.answer}")

            chat_history.add_data(conversation_id, f"Question: {question}")
            chat_history.add_data(conversation_id, f"Answer: {response.answer}")
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")

import os
import sys
import uuid
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.memory.local_memory import LocalMemory
from app.rag_engine.indexing_pipeline import IndexingPipeline
from app.rag_engine.retrieval_pipeline import RetrievalPipeline
from app.planner_agent import PlannerAgent


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


if __name__ == "__main__":
    setup_logging()

    print("📒 Travel Journal RAG Assistant (type 'exit' to quit)\n")

    _ = IndexingPipeline.add_trip_to_vector_store()
    logging.info("Trip data indexed successfully.")

    chat_history = LocalMemory()
    planner_agent = PlannerAgent()
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
                response = planner_agent.run(user_query=question)
            except Exception as e:
                logging.error(f"Error during retrieval: {e}")
                print(
                    "⚠️ An error occurred while processing your request. Please try again."
                )
                continue

            if response.final_answer is None:
                print("❓ No answer found. Please try a different question.")
                continue
            print(f"💬 Answer: {response.final_answer}")

            chat_history.add_data(conversation_id, f"Question: {question}")
            chat_history.add_data(conversation_id, f"Answer: {response.final_answer}")
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")

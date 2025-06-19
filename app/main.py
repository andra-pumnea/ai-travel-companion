import os
import sys
import uuid
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.engine.memory.in_memory_history import InMemoryHistory
from app.engine.indexing_pipeline import IndexingPipeline
from app.engine.retrieval_pipeline import RetrievalPipeline


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

    chat_history = InMemoryHistory()
    conversation_id = uuid.uuid4().hex

    try:
        while True:
            question = input("\n❓ Ask a question: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            if question == "":
                continue

            response = RetrievalPipeline.run_retrieval_pipeline(
                user_query=question,
                prompt_name="question_answering",
                conversation_id=conversation_id,
            )
            print(f"💬 Answer: {response.answer}")
            print(f"📄 Thought process: {response.thought_process}\n")

            chat_history.add_data(conversation_id, f"Question: {question}")
            chat_history.add_data(conversation_id, f"Answer: {response.answer}")
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")

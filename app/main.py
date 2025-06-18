import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

    chat_history = []

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
                chat_history=chat_history,
            )
            print(f"💬 Answer: {response.answer}")
            print(f"📄 Thought process: {response.thought_process}\n")

            chat_history.append(
                {"role": "assistant", "content": response.answer},
            )
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")

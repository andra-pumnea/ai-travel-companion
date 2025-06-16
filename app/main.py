import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.engine.indexing_pipeline import add_trip_to_vector_store
from app.engine.retrieval_pipeline import run_retrieval_pipeline

if __name__ == "__main__":
    print("📒 Travel Journal RAG Assistant (type 'exit' to quit)\n")

    _ = add_trip_to_vector_store()
    print("Trip added to vector store successfully.")

    try:
        while True:
            question = input("\n❓ Ask a question: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            if question == "":
                continue

            response = run_retrieval_pipeline(
                user_query=question, prompt_name="summarize_trip"
            )
            print(f"💬 Answer: {response}")
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")

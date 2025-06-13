import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    add_start_index=True,
)

if __name__ == "__main__":
    from app.data_management.data_loader import (
        read_trip_from_polarsteps,
        convert_trip_to_documents,
    )

    trip = read_trip_from_polarsteps()
    documents = convert_trip_to_documents(trip)

    all_splits = []
    for idx, doc in enumerate(documents):
        chunks = text_splitter.split_text(doc.page_content)
        if len(chunks) > 1:
            print(
                f"Step {idx}: {doc.metadata['display_name']} split into chunks: {chunks}"
            )
        print(f"Step: {idx}, Number of chunks: {len(chunks)}")

        all_splits.extend(chunks)

    print(f"Total number of documents: {len(documents)}")
    print(f"Total number of splits: {len(all_splits)}")

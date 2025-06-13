import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
from langchain_core.documents import Document


from app.data_models.trip import Trip, TripStep


def read_trip_from_polarsteps() -> Trip:
    with open(os.path.expanduser("~/ai-travel-companion/data/trip.json"), "r") as f:
        data = json.load(f)
    trip = Trip(**data)
    return trip


def pydantic_to_document(trip_step: TripStep) -> Document:
    return Document(
        page_content=trip_step.description or "",
        metadata={
            "id": trip_step.id,
            "display_name": trip_step.display_name,
            "location": {
                "lat": trip_step.location.lat,
                "lon": trip_step.location.lon,
                "name": trip_step.location.name,
                "detail": trip_step.location.detail,
                "country_code": trip_step.location.country_code,
            },
        },
    )


def convert_trip_to_documents(trip: Trip) -> list[Document]:
    documents = []
    for step in trip.all_steps:
        document = pydantic_to_document(step)
        documents.append(document)
    return documents


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    trip = read_trip_from_polarsteps()
    documents = convert_trip_to_documents(trip)
    for idx, doc in enumerate(documents):
        if len(doc.page_content) == 0:
            print(
                f"Step {idx} has no content. Step name: {doc.metadata['display_name']}"
            )
        else:
            token_counts = [
                len(tokenizer.tokenize(doc.page_content)) for doc in documents
            ]
            print(f"Index: {idx}, Total characters: {len(doc.page_content)}")

    average_tokens = sum(token_counts) / len(token_counts)
    print(f"Average tokens per document: {average_tokens:.2f}")

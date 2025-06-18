import os
import sys
import json
from typing import Type
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data_models.trip import Trip
from app.prompts.prompt_manager import PromptManager
from app.prompts.prompt_responses import QAResponse
from app.engine.llm_clients.llm_client_factory import LLMClientFactory

client = LLMClientFactory(provider="groq")


def read_trip_from_polarsteps() -> Trip:
    with open(os.path.expanduser("~/ai-travel-companion/data/trip.json"), "r") as f:
        data = json.load(f)
    trip = Trip(**data)
    return trip


def call_llm(
    user_query: str, prompt, response_model: Type[BaseModel], max_tokens: int = 400
) -> str:
    response = client.create_completion(
        response_model=response_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ],
        max_tokens=max_tokens,
    )
    return response


if __name__ == "__main__":
    # user_query = (
    #     "Make an itinerary for someone visiting Japan and Philippines in one week."
    # )
    # prompt = PromptManager.get_prompt("plan_trip")
    # response = call_llm(user_query, prompt)
    # print(response)

    trip = read_trip_from_polarsteps()
    prompt = PromptManager.get_prompt(
        "question_answering",
        context=trip.all_steps[4].description,
    )
    user_query = "where did I got in japan"
    response = call_llm(user_query, prompt, response_model=QAResponse, max_tokens=200)
    print(response)

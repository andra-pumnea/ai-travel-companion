import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

from app.data_models.trip import Trip
from app.prompts.prompt_manager import PromptManager
from app.llm_manager import LLMManager


def read_trip_from_polarsteps() -> Trip:
    with open(os.path.expanduser("~/ai-travel-companion/data/trip.json"), "r") as f:
        data = json.load(f)
    trip = Trip(**data)
    return trip


def run_llm_manager(user_query: str, prompt: str = None, max_tokens: int = 400) -> str:
    llm_manager = LLMManager()
    response = llm_manager.generate_response(
        user_query=user_query, prompt=prompt, max_tokens=max_tokens
    )
    return response


if __name__ == "__main__":
    user_query = (
        "Make an itinerary for someone visiting Japan and Philippines in one week."
    )
    prompt = PromptManager.get_prompt("plan_trip")
    response = run_llm_manager(user_query, prompt)
    print(response)

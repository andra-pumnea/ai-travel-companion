import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groq import Groq
import json

from app.config import LLMConfig, GroqConfig
from app.data_models.trip import Trip
from app.prompts.prompt_manager import PromptManager

llm_config = LLMConfig()
groq_config = GroqConfig()
client = Groq(
    api_key=groq_config.groq_api_key,
)


def read_trip_from_polarsteps() -> Trip:
    with open(os.path.expanduser("~/ai-travel-companion/data/trip.json"), "r") as f:
        data = json.load(f)
    trip = Trip(**data)
    return trip


def call_llm(user_query: str, prompt, max_tokens: int = 400) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        model=llm_config.model,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    # Print the completion returned by the LLM.
    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    # user_query = (
    #     "Make an itinerary for someone visiting Japan and Philippines in one week."
    # )
    # prompt = PromptManager.get_prompt("plan_trip")
    # response = call_llm(user_query, prompt)
    # print(response)

    trip = read_trip_from_polarsteps()
    additional_instruction = "Summarize using only emojis."
    prompt = PromptManager.get_prompt(
        "summarize_trip",
        travellers="Andrada, Pablo",
        is_additional_instruction=True,
        additional_instruction=additional_instruction,
    )
    user_query = trip.all_steps[4].description
    response = call_llm(user_query, prompt, max_tokens=200)
    print(response)

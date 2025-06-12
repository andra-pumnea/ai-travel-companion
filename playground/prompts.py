import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.prompts.prompt_manager import PromptManager

plan_trip_prompt = PromptManager.get_prompt("plan_trip")
print(plan_trip_prompt)

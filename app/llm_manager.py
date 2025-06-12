from app.config import LLMConfig
from app.engine.llm_clients.llm_client_factory import get_llm_client


class LLMManager:
    def __init__(self):
        self.llm_config = LLMConfig()
        self.client = get_llm_client(self.llm_config)

    def generate_response(self, user_query: str, prompt, max_tokens: int = 400) -> str:
        """
        Calls the LLM with the provided user query and prompt.
        :param user_query: The query from the user.
        :param prompt: The prompt to be used for the LLM.
        :param max_tokens: The maximum number of tokens to generate in the response.
        :return: The response from the LLM.
        """
        chat_completion = self.client.chat.completions.create(
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
            model=self.llm_config.model,
            temperature=0.0,
            max_tokens=max_tokens,
        )

        return chat_completion.choices[0].message.content

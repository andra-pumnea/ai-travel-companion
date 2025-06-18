import logging

from app.engine.vector_store import VectorStore
from app.prompts.prompt_manager import PromptManager
from app.engine.llm_manager import LLMManager
from app.prompts.prompt_responses import (
    QAResponse,
    CountryExtractionResponse,
    RewriteQueryResponse,
)


class RetrievalPipeline:
    _vector_store = VectorStore()
    _llm_manager = LLMManager()

    @staticmethod
    def retrieve(user_query: str, metadata: dict = None):
        retrieved_docs = RetrievalPipeline._vector_store.similarity_search(
            query=user_query, metadata=metadata, k=10
        )
        logging.info(
            f"Retrieved {len(retrieved_docs)} documents for query: {user_query}"
        )
        return {"context": retrieved_docs}

    @staticmethod
    def generate(user_query: str, prompt: str, chat_history: list, response_model=None):
        response = RetrievalPipeline._llm_manager.generate_response(
            user_query=user_query,
            prompt=prompt,
            chat_history=chat_history,
            response_model=response_model,
        )
        return response

    @staticmethod
    def run_retrieval_pipeline(user_query: str, prompt_name: str, chat_history: list):
        """
        Runs the retrieval pipeline to get a response based on the user query and prompt.

        :param user_query: The query from the user.
        :param prompt_name: The name of the prompt to be used.
        :return: The generated response from the LLM.
        """
        # Rewrite the user query if necessary
        if chat_history:
            rewrite_prompt = PromptManager.get_prompt(
                "rewrite_user_query",
                **{
                    "previous_answer": chat_history[-1]["content"],
                    "followup_question": user_query,
                },
            )
            rewrite_query_response = RetrievalPipeline.generate(
                user_query,
                rewrite_prompt,
                chat_history=chat_history,
                response_model=RewriteQueryResponse,
            )
            logging.info(
                f"Rewritten user query: {rewrite_query_response.rewritten_user_query}"
            )
            user_query = rewrite_query_response.rewritten_user_query

        # Extract country code from the user query
        country_extraction_prompt = PromptManager.get_prompt("country_extraction")
        country_code_response = RetrievalPipeline.generate(
            user_query,
            country_extraction_prompt,
            chat_history=[],
            response_model=CountryExtractionResponse,
        )
        logging.info(f"Extracted country code: {country_code_response.country_code}")
        metadata = {
            "country_code": country_code_response.country_code
            if country_code_response.country_code
            else ""
        }

        # Retrieve documents from the vector store based on the user query and metadata
        docs = RetrievalPipeline.retrieve(user_query, metadata=metadata)
        context = (
            "\n\n".join(doc.page_content for doc in docs["context"]) if docs else ""
        )

        # Generate the response using the LLM with the retrieved context
        chat_prompt = PromptManager.get_prompt(prompt_name, **{"context": context})
        response = RetrievalPipeline.generate(
            user_query, chat_prompt, chat_history, response_model=QAResponse
        )
        return response

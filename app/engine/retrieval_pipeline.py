import logging
from typing import Type
from pydantic import BaseModel

from app.engine.vector_store import VectorStore
from app.prompts.prompt_manager import PromptManager
from app.engine.llm_manager import LLMManager
from app.prompts.prompt_responses import (
    QAResponse,
    CountryExtractionResponse,
    QueryRewritingResponse,
)
from app.engine.memory.in_memory_history import InMemoryHistory


class RetrievalPipeline:
    _vector_store = VectorStore()
    _llm_manager = LLMManager()
    _memory = InMemoryHistory()

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
    def generate(
        user_query: str,
        prompt: str,
        conversation_id: str,
        response_model: Type[BaseModel],
    ):
        response = RetrievalPipeline._llm_manager.generate_response(
            user_query=user_query,
            prompt=prompt,
            conversation_id=conversation_id,
            response_model=response_model,
        )
        return response

    @staticmethod
    def run_retrieval_pipeline(user_query: str, prompt_name: str, conversation_id: str):
        """
        Runs the retrieval pipeline to get a response based on the user query and prompt.

        :param user_query: The query from the user.
        :param prompt_name: The name of the prompt to be used.
        :return: The generated response from the LLM.
        """
        # Rewrite the user query if necessary
        if RetrievalPipeline._memory.get_data(conversation_id):
            rewrite_prompt = PromptManager.get_prompt(
                "query_rewriting",
                **{
                    "conversation_history": RetrievalPipeline._memory.get_data(
                        conversation_id
                    )[-5:],
                    "followup_question": user_query,
                },
            )
            rewrite_query_response = RetrievalPipeline.generate(
                user_query,
                rewrite_prompt,
                conversation_id,
                response_model=QueryRewritingResponse,
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
            conversation_id,
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
            user_query, chat_prompt, conversation_id, response_model=QAResponse
        )
        return response

import logging
from typing import Any

from app.rag_engine.vector_store import VectorStore
from app.llms.llm_manager import LLMManager
from app.memory.conversation_history.local_memory import LocalMemory
from app.prompts.query_rewriting import QueryRewriting
from app.prompts.question_answering import QuestionAnswering


class RetrievalPipeline:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, vector_store: VectorStore):
        if not self._initialized:
            self.vector_store = vector_store
            self.llm_manager = LLMManager()
            self.memory = LocalMemory()
            self._initialized = True

    @staticmethod
    def _log_token_usage(prompt_name: str, prompt: str) -> None:
        """Logs the token usage for a given prompt.
        :param prompt_name: The name of the prompt.
        :param prompt: The rendered prompt string."""
        logging.info(f"Prompt {prompt_name} token usage: {len(prompt)}")

    def search_journal_entries(
        self, user_query: str, user_trip_id: str, limit: int = 5
    ) -> list[dict]:
        """
        Retrieves relevant documents from the vector store based on the user query.
        :param user_query: The query from the user.
        :param metadata: Optional metadata to filter the search.
        :return: A list of retrieved documents."""
        collection_name = f"{user_trip_id}_trip_collection"
        retrieved_docs = self.vector_store.search(
            query=user_query, collection_name=collection_name, limit=limit
        )
        logging.info(
            f"Retrieved {len(retrieved_docs)} documents for query: {user_query}"
        )
        return retrieved_docs

    def get_all_journal_entries(self, user_trip_id: str) -> list[dict]:
        """
        Retrieves all journal entries for a given trip.
        :param user_trip_id: The ID of the user's trip.
        :return: A list of all journal entries."""
        collection_name = f"{user_trip_id}_trip_collection"
        all_docs = self.vector_store.get_all_documents(collection_name=collection_name)
        logging.info(f"Retrieved {len(all_docs)} documents for trip ID: {user_trip_id}")
        return all_docs

    def _rewrite_query(
        self, user_query: str, conversation_id: str, max_history: int = 5
    ) -> str:
        """
        Rewrites the user query based on the conversation history.
        :param user_query: The original user query.
        :param conversation_id: The ID of the conversation.
        :return: The rewritten user query.
        """
        memory_data = self.memory.get_history(conversation_id)
        if memory_data:
            messages = [msg.model_dump() for msg in memory_data[-max_history:]]
            rendered_prompt = QueryRewriting.format(
                conversation_history=messages[-max_history:],
                followup_question=user_query,
            )
            RetrievalPipeline._log_token_usage(
                prompt_name=QueryRewriting.prompt_name, prompt=rendered_prompt
            )

            rewrite_query_response = self.llm_manager.call_llm_with_retry(
                user_query=user_query,
                prompt=rendered_prompt,
                response_model=QueryRewriting.response_model(),
            )
            logging.info(
                f"Rewritten user query: {rewrite_query_response.rewritten_user_query}"
            )
            return rewrite_query_response.rewritten_user_query
        return user_query

    def _generate_answer(self, user_query: str, context: str) -> Any:
        """
        Generates an answer based on the user query and context.
        :param user_query: The original user query.
        :return: The generated answer.
        """
        rendered_prompt = QuestionAnswering.format(context=context)

        RetrievalPipeline._log_token_usage(
            prompt_name=QuestionAnswering.prompt_name, prompt=rendered_prompt
        )

        response = self.llm_manager.call_llm_with_retry(
            user_query=user_query,
            prompt=rendered_prompt,
            response_model=QuestionAnswering.response_model(),
        )
        return response

    def search_with_generation(
        self, user_query: str, user_trip_id: str, limit: int = 5
    ) -> tuple[list[dict], Any]:
        """
        Runs the retrieval pipeline to get a response based on the user query and prompt.

        :param user_query: The query from the user.
        :param prompt_name: The name of the prompt to be used.
        :return: The generated response from the LLM.
        """
        user_query = self._rewrite_query(user_query, user_trip_id)

        docs = self.search_journal_entries(user_query, user_trip_id, limit)
        context = "\n\n".join(doc["description"] for doc in docs) if docs else ""

        response = self._generate_answer(user_query=user_query, context=context)
        return response.answer, docs

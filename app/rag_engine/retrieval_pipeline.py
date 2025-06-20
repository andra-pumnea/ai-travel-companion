import logging
from typing import Any

from app.rag_engine.vector_store import VectorStore
from app.llms.llm_manager import LLMManager
from app.rag_engine.memory.local_memory import LocalMemory
from app.prompts.query_rewriting import QueryRewriting
from app.prompts.question_answering import QuestionAnswering


class RetrievalPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_manager = LLMManager()
        self.memory = LocalMemory()

    @staticmethod
    def _log_token_usage(prompt_name: str, prompt: str) -> None:
        """Logs the token usage for a given prompt.
        :param prompt_name: The name of the prompt.
        :param prompt: The rendered prompt string."""
        logging.info(f"Prompt {prompt_name} token usage: {len(prompt)}")

    def _search_journal_entries(self, user_query: str, metadata: dict = None) -> dict:
        """
        Retrieves relevant documents from the vector store based on the user query.
        :param user_query: The query from the user.
        :param metadata: Optional metadata to filter the search.
        :return: A list of retrieved documents."""
        retrieved_docs = self.vector_store.similarity_search(
            query=user_query, metadata=metadata, k=10
        )
        logging.info(
            f"Retrieved {len(retrieved_docs)} documents for query: {user_query}"
        )
        return {"context": retrieved_docs}

    def _rewrite_query(self, user_query: str, conversation_id: str) -> str:
        """
        Rewrites the user query based on the conversation history.
        :param user_query: The original user query.
        :param conversation_id: The ID of the conversation.
        :return: The rewritten user query.
        """
        memory_data = self.memory.get_data(conversation_id)
        if memory_data:
            rendered_prompt = QueryRewriting.format(
                conversation_history=memory_data[-5:],
                followup_question=user_query,
            )
            RetrievalPipeline._log_token_usage(
                prompt_name=QueryRewriting.prompt_name, prompt=rendered_prompt
            )

            rewrite_query_response = self.llm_manager.call_llm_with_retry(
                user_query=user_query,
                prompt=rendered_prompt,
                conversation_id=conversation_id,
                response_model=QueryRewriting.response_model(),
            )
            logging.info(
                f"Rewritten user query: {rewrite_query_response.rewritten_user_query}"
            )
            return rewrite_query_response.rewritten_user_query
        return user_query

    def _generate_answer(
        self, user_query: str, context: str, conversation_id: str
    ) -> Any:
        """
        Generates an answer based on the user query and context.
        :param user_query: The original user query.
        :param context: The context retrieved from the vector store.
        :param conversation_id: The ID of the conversation.
        :return: The generated answer.
        """
        rendered_prompt = QuestionAnswering.format(context=context)

        RetrievalPipeline._log_token_usage(
            prompt_name=QuestionAnswering.prompt_name, prompt=rendered_prompt
        )

        response = self.llm_manager.call_llm_with_retry(
            user_query=user_query,
            prompt=rendered_prompt,
            conversation_id=conversation_id,
            response_model=QuestionAnswering.response_model(),
        )
        return response

    def run(self, user_query: str, conversation_id: str):
        """
        Runs the retrieval pipeline to get a response based on the user query and prompt.

        :param user_query: The query from the user.
        :param prompt_name: The name of the prompt to be used.
        :return: The generated response from the LLM.
        """
        # Rewrite the user query if necessary
        user_query = self._rewrite_query(user_query, conversation_id)

        # Retrieve documents from the vector store based on the user query and metadata
        docs = self._search_journal_entries(user_query)
        context = (
            "\n\n".join(doc.page_content for doc in docs["context"]) if docs else ""
        )

        # Generate the response using the LLM with the retrieved context
        response = self._generate_answer(
            user_query=user_query,
            context=context,
            conversation_id=conversation_id,
        )
        return response

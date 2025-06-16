from app.engine.vector_store import VectorStore
from app.prompts.prompt_manager import PromptManager
from app.llm_manager import LLMManager


def retrieve(user_query: str):
    vector_store = VectorStore(collection_name="trip_collection").get_vector_store()
    retrieved_docs = vector_store.similarity_search(query=user_query, k=5)
    print(f"Retrieved {len(retrieved_docs)} documents for query: {user_query}")
    print(f"Retrieved documents: {retrieved_docs}")
    return {"context": retrieved_docs}


def generate(user_query: str, prompt: str):
    llm_manager = LLMManager()
    response = llm_manager.generate_response(user_query=user_query, prompt=prompt)
    return response


def run_retrieval_pipeline(user_query: str, prompt_name: str):
    """
    Runs the retrieval pipeline to get a response based on the user query and prompt.

    :param user_query: The query from the user.
    :param prompt_name: The name of the prompt to be used.
    :param prompt_input: Optional input for the prompt.
    :return: The generated response from the LLM.
    """
    docs = retrieve(user_query)
    context = "\n\n".join(doc.page_content for doc in docs["context"]) if docs else ""
    chat_prompt = PromptManager.get_prompt(prompt_name, **{"context": context})
    print(f"Using prompt: \n {chat_prompt}")
    response = generate(user_query, chat_prompt)
    return response

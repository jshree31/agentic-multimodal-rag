from langchain.tools import tool
from core.db import get_vector_store


@tool
def vector_search(query: str, k: int = 5):
    """
    Semantic vector similarity search.
    Best for conceptual or paraphrased queries.
    """
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "retrieval_type": "vector",
        }
        for doc in docs
    ]
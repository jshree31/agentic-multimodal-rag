from langchain.tools import tool
from src.core.db import vector_search_db

@tool
def vector_search(query: str, k: int = 5, chunk_type: str | None = None):
    """Semantic vector similarity search."""
    return vector_search_db(query, k=k, chunk_type=chunk_type)
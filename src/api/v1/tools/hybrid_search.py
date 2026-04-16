from langchain.tools import tool
from src.core.db import hybrid_search_db

@tool
def hybrid_search(query: str, k: int = 5, chunk_type: str | None = None):
    """Hybrid search (vector + FTS) using RRF."""
    return hybrid_search_db(query, k=k, chunk_type=chunk_type)
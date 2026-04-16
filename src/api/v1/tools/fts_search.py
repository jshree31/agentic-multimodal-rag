from langchain.tools import tool
from src.core.db import fts_search_db

@tool
def fts_search(query: str, k: int = 5, chunk_type: str | None = None):
    """Full-text keyword search using PostgreSQL FTS on multimodal_chunks."""
    return fts_search_db(query, k=k, chunk_type=chunk_type)
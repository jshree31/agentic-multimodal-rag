from langchain.tools import tool
from core.db import get_vector_store
from api.v1.tools.fts_search import fts_search


@tool
def hybrid_search(query: str, k: int = 5):
    """
    Hybrid search combining vector and FTS results using RRF.
    """
    vector_store = get_vector_store()
    vector_docs = vector_store.similarity_search(query, k=k)
    fts_docs = fts_search(query, k=k)

    rrf_scores = {}
    chunk_map = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "retrieval_type": "hybrid",
        }

    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = {
            "content": item["content"],
            "metadata": item["metadata"],
            "retrieval_type": "hybrid",
        }

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[key] for key, _ in ranked[:k]]
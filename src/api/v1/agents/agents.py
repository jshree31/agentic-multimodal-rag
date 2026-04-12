from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict):
    # User input
    original_query: str
    current_query: str

    # Loop control
    iteration_count: int
    max_iterations: int

    # Retrieval
    retrieval_strategy: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]

    # Reranking
    reranked_chunks: List[Dict[str, Any]]

    # Validation
    is_relevant: Optional[bool]
    refined_query: Optional[str]

    # Final output
    final_answer: Optional[str]
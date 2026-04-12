from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END


# 1. State Definition
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


# 2. Nodes definition
def retrieval_agent(state: GraphState) -> GraphState:
    """
    Retrieves documents for the current query.
    Dynamically selects vector, FTS, or hybrid (RRF) search using LLM tools.
    """
    return state

def rerank_node(state: GraphState) -> GraphState:
    """
    Reranks retrieved chunks to improve relevance and precision.
    Produces a refined ordering of candidate documents.
    """
    return state

def validate_agent(state: GraphState) -> GraphState:
    """
    Validates whether the reranked chunks sufficiently answer the query.
    Triggers query refinement if relevance is insufficient.
    """
    return state

def generate_answer_agent(state: GraphState) -> GraphState:
    """
    Generates the final user-readable answer from validated knowledge.
    Formats and synthesizes the response.
    """
    return state


# Routing function
def should_retry_or_continue(state: GraphState) -> str:
    if (
        state["is_relevant"] is False
        and state["iteration_count"] < state["max_iterations"]
    ):
        return "retry"
    return "continue"


# Build the langGraph
def build_query_graph():
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("rerank", rerank_node)
    graph.add_node("validate", validate_agent)
    graph.add_node("generate_answer", generate_answer_agent)

    # Entry point
    graph.set_entry_point("retrieval")

    # Linear flow
    graph.add_edge("retrieval", "rerank")
    graph.add_edge("rerank", "validate")

    # Conditional loop
    graph.add_conditional_edges(
        "validate",
        should_retry_or_continue,
        {
            "retry": "retrieval",
            "continue": "generate_answer",
        },
    )

    # End
    graph.add_edge("generate_answer", END)

    return graph.compile()
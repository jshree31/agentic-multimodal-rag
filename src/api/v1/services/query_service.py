from typing import Dict, Any, Optional
from src.api.v1.agents.agents import build_query_graph, GraphState


# ─────────────────────────────────────────────────────────────
# LangGraph initialization (compile once per process)
# ─────────────────────────────────────────────────────────────
_graph = build_query_graph()


# ─────────────────────────────────────────────────────────────
# Public API: Query Execution Service
# ─────────────────────────────────────────────────────────────
def run_query(
    query: str,
    k: int = 5,
    chunk_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the agentic LangGraph workflow and returns
    route-specific formatted responses.

    Routes:
    - document
    - banking
    - hybrid
    """

    # ── Initial State ────────────────────────────────────────
    initial_state: GraphState = {
        "original_query": query,
        "current_query": query,

        "normalized_query": None,
        "is_query_valid": None,
        "route": None,

        "iteration_count": 0,
        "max_iterations": 3,

        "retrieval_strategy": None,
        "retrieved_chunks": [],
        "reranked_chunks": [],

        "is_relevant": None,
        "refined_query": None,

        "generated_sql": None,
        "sql_result": None,

        "final_answer": None,
    }

    # ── Run LangGraph ─────────────────────────────────────────
    final_state: GraphState = _graph.invoke(initial_state)

    route = final_state.get("route")
    iterations = final_state.get("iteration_count") or 0
    search_type = final_state.get("retrieval_strategy")
    final_answer = final_state.get("final_answer") or {}

    # ─────────────────────────────────────────────────────────
    # DOCUMENT RESPONSE
    # ─────────────────────────────────────────────────────────
    if route == "document":
        return {
            "route": route,
            "query": query,
            "answer": final_answer.get("answer", "No answer generated."),
            "relevant_chunks": final_answer.get("sources", []),
            "iterations": iterations,
            "search_type": search_type,
            "policy_citations": final_answer.get(
                "policy_citations",
                "Derived from internal policy documents"
            ),
        }

    # ─────────────────────────────────────────────────────────
    # BANKING RESPONSE
    if route == "banking":
        return {
            "route": route,
            "query": query,

            # LLM explanation
            "answer": final_answer.get("answer", "No answer generated."),

            # ✅ RAW SQL
            "sql_query_executed": final_state.get("generated_sql"),

            # ✅ RAW SQL RESULT
            "sql_result": final_state.get("sql_result"),

            "iterations": iterations,
            "database_name": "NorthStar Bank Core Banking Database",
        }

    # ─────────────────────────────────────────────────────────
    # HYBRID RESPONSE
    # ─────────────────────────────────────────────────────────
    if route == "hybrid":
        return {
            "route": route,
            "query": query,
            "answer": final_answer.get("answer", "No answer generated."),
            "relevant_chunks": final_answer.get("sources", []),
            "iterations": iterations,
            "search_type": search_type,
            "policy_citations": final_answer.get(
                "policy_citations",
                "Derived from internal policy documents"
            ),
            "database_name": "NorthStar Bank Core Banking Database",
            "sql_query_executed": final_state.get("generated_sql"),
            "banking_data": final_state.get("sql_result"),
        }

    # ─────────────────────────────────────────────────────────
    # FALLBACK
    # ─────────────────────────────────────────────────────────
    return {
        "route": route,
        "query": query,
        "answer": "Unable to determine response type.",
        "iterations": iterations,
    }
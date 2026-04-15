import os
import re
from typing import TypedDict, List, Dict, Any, Optional, Literal

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
import cohere

from src.api.v1.tools.vector_search import vector_search
from src.api.v1.tools.fts_search import fts_search
from src.api.v1.tools.hybrid_search import hybrid_search
from src.core.db import get_sql_database
from src.api.v1.schemas.query_schema import AIResponse


# ─────────────────────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=0,
)

llm_with_tools = llm.bind_tools([
    vector_search,
    fts_search,
    hybrid_search,
])


# ─────────────────────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    original_query: str
    current_query: str

    normalized_query: Optional[str]
    is_query_valid: Optional[bool]

    route: Optional[str]   # document | banking | hybrid

    iteration_count: int
    max_iterations: int

    retrieval_strategy: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]

    is_relevant: Optional[bool]
    refined_query: Optional[str]

    generated_sql: Optional[str]
    sql_result: Optional[Any]

    final_answer: Optional[Dict[str, Any]]


# ─────────────────────────────────────────────────────────────
# Node 1: Query Validation (HyDE)
# ─────────────────────────────────────────────────────────────
def query_validation_agent(state: GraphState) -> GraphState:
    system_prompt = (
        "You are a query validation agent.\n"
        "If the query is clear, respond with:\n"
        "VALID: <same query>\n\n"
        "If vague, respond with:\n"
        "REWRITE: <clear, retrieval-ready query>"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["original_query"]),
    ]

    response = llm.invoke(messages)
    content = response.content

    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))

    content = content.strip()

    if content.startswith("REWRITE:"):
        rewritten = content.replace("REWRITE:", "").strip()
        state["current_query"] = rewritten
        state["normalized_query"] = rewritten
        state["is_query_valid"] = False
    else:
        state["current_query"] = state["original_query"]
        state["normalized_query"] = None
        state["is_query_valid"] = True

    return state


# ─────────────────────────────────────────────────────────────
# Node 2: Router (Document | Banking | Hybrid)
# ─────────────────────────────────────────────────────────────
class _RouteDecision(BaseModel):
    route: Literal["document", "banking", "hybrid"]
    reason: str


def router_node(state: GraphState) -> GraphState:
    structured_llm = llm.with_structured_output(_RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a query router for a Smart Banking Assistant.

Classify the query into EXACTLY one route:

"banking" —
Queries that ONLY require structured banking data such as:
balances, transactions, spending, loans, EMIs,
credit card dues, fixed deposit amounts, account details.

"document" —
Queries that ONLY require product rules, eligibility,
fees, interest rates, RBI regulations, policies,
and internal knowledge-base documents.

"hybrid" —
Queries that require BOTH:
- personal banking data (my account, my loan, my card, my balance)
AND
- rules, explanations, or regulatory context
(charges, eligibility, RBI policies, why something happened).

Reply ONLY with the route and one short reason.
"""),
        ("human", "Query: {current_query}")
    ])

    chain = prompt | structured_llm
    decision = chain.invoke({"current_query": state["current_query"]})

    print(f"[router_node] Route = {decision.route} | {decision.reason}")
    state["route"] = decision.route
    return state


# ─────────────────────────────────────────────────────────────
# Node 3: NL2SQL (Banking RDBMS)
# ─────────────────────────────────────────────────────────────
def nl2sql_node(state: GraphState) -> GraphState:
    """
    Executes NL2SQL over the NorthStar Bank Core Banking Database.

    Responsibilities:
    - Generate a safe SELECT-only SQL query
    - Execute SQL on read-only banking DB
    - Summarize results using structured AIResponse
    - Populate GraphState for banking / hybrid flows
    """

    # ── Iteration semantics ───────────────────────────────────
    # Banking / Hybrid counts as exactly one reasoning iteration
    if state["iteration_count"] == 0:
        state["iteration_count"] = 1

    # ── Database connection ───────────────────────────────────
    db = get_sql_database()

    # ── NL2SQL prompt (banking-tuned) ─────────────────────────
    sql_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a PostgreSQL expert for a retail banking database.

Generate EXACTLY ONE safe SELECT query to answer the user's question.

Rules:
- Output ONLY raw SQL (no markdown, no explanations)
- NEVER generate INSERT, UPDATE, DELETE, DROP, or ALTER
- ALWAYS include a LIMIT clause (maximum 50 rows)
- Use ONLY tables and columns present in the schema
- Ignore currency symbols (₹, commas) and treat all amounts as numeric INR
- If the question mentions time ranges (e.g. last 3 months),
  translate them using CURRENT_DATE - INTERVAL syntax
- Prefer account_id as the primary identifier when available

Database schema:
{schema}
"""
        ),
        ("human", "Question: {question}")
    ])

    # ── Step 1: Generate SQL ──────────────────────────────────
    sql_chain = sql_prompt | llm
    raw_sql = sql_chain.invoke({
        "schema": db.get_table_info(),
        "question": state["current_query"]
    })

    # Safe extraction (Gemini may return list)
    sql = raw_sql.content
    if isinstance(sql, list):
        sql = " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in sql
        )

    sql = sql.strip().strip("```")

    # Optional (debug only)
    # print("[nl2sql_node] Generated SQL:")
    # print(sql)

    # ── Step 2: Execute SQL ───────────────────────────────────
    try:
        sql_result = db.run(sql)
    except Exception:
        sql_result = (
            "The requested banking information could not be retrieved "
            "due to a database error."
        )

    # Optional (debug only)
    # print("[nl2sql_node] SQL Result:")
    # print(sql_result)

    # ── Step 3: Summarize using structured output ─────────────
    structured_llm = llm.with_structured_output(AIResponse)

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a banking data analyst. "
            "Answer the user's question using ONLY the SQL results provided. "
            "Summarize clearly and concisely. "
            "If multiple records exist, group them logically."
        ),
        (
            "human",
            "Question: {query}\n\n"
            "SQL Executed:\n{sql}\n\n"
            "SQL Results:\n{result}"
        )
    ])

    chain = answer_prompt | structured_llm

    answer = chain.invoke({
        "query": state["current_query"],
        "sql": sql,
        "result": sql_result
    })

    # ── Step 4: Update GraphState ─────────────────────────────
    state["generated_sql"] = sql
    state["sql_result"] = sql_result
    state["final_answer"] = answer.model_dump()

    return state

# ─────────────────────────────────────────────────────────────
# Node 4: Retrieval (Documents)
# ─────────────────────────────────────────────────────────────
def retrieval_agent(state: GraphState) -> GraphState:
    response = llm_with_tools.invoke([
        SystemMessage(content="Choose the best retrieval tool."),
        HumanMessage(content=state["current_query"]),
    ])

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_fn = {
            "vector_search": vector_search,
            "fts_search": fts_search,
            "hybrid_search": hybrid_search,
        }[tool_call["name"]]

        results = tool_fn.invoke(tool_call["args"])
        state["retrieved_chunks"] = results
        state["retrieval_strategy"] = tool_call["name"]
    else:
        state["retrieved_chunks"] = []
        state["retrieval_strategy"] = None

    return state


# ─────────────────────────────────────────────────────────────
# Node 5: Rerank
# ─────────────────────────────────────────────────────────────
def rerank_node(state: GraphState) -> GraphState:
    docs = state.get("retrieved_chunks", [])

    if not docs:
        state["reranked_chunks"] = []
        return state

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    res = co.rerank(
        model="rerank-english-v3.0",
        query=state["current_query"],
        documents=[d["content"] for d in docs],
        top_n=min(10, len(docs))
    )

    reranked = []
    for r in res.results:
        doc = docs[r.index]
        doc["confidence"] = round(float(r.relevance_score), 3)
        reranked.append(doc)

    state["reranked_chunks"] = reranked
    return state


# ─────────────────────────────────────────────────────────────
# Node 6: Validate & Retry (only on empty recall)
# ─────────────────────────────────────────────────────────────
def validate_agent(state: GraphState) -> GraphState:
    state["iteration_count"] += 1

    if not state.get("reranked_chunks"):
        state["is_relevant"] = False

        prompt = (
            "Rewrite the query to be more general but intent-preserving.\n"
            "Output ONLY the rewritten query."
        )
        resp = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state["current_query"]),
        ])

        content = resp.content
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content)

        state["current_query"] = content.strip()
        return state

    state["is_relevant"] = True
    return state


# ─────────────────────────────────────────────────────────────
# Node 7: Generate Answer (Document + Optional SQL Context)
# ─────────────────────────────────────────────────────────────
def generate_answer_agent(state: GraphState) -> GraphState:
    chunks = state.get("reranked_chunks", [])
    sources = []
    policy_refs = set()
    text_blocks = []

    sql_context = ""
    if state.get("sql_result"):
        sql_context = f"\n\nBanking Data:\n{state['sql_result']}"

    for c in chunks:
        meta = c.get("metadata", {})
        page = meta.get("page_number") or meta.get("page") or "N/A"
        section = meta.get("section", "—")

        matches = re.findall(
            r"(RBI\/\d{4}-\d{2}\/\d+|RBI Circular [^\.]+)",
            c["content"]
        )

        for m in matches:
            policy_refs.add(m)

        sources.append({
            "content": c["content"][:800],
            "page": page,
            "section": section,
            "confidence_score": c.get("confidence"),
            "source_file": meta.get("source_file"),
        })

        text_blocks.append(
            f"[Page {page} | {section}]\n{c['content']}"
        )

    policy_citations = (
        ", ".join(sorted(policy_refs))
        if policy_refs else "Derived from internal policy documents"
    )

    response = llm.invoke([
        SystemMessage(content="Answer ONLY using the provided context."),
        HumanMessage(
            content=(
                sql_context +
                "\n\n".join(text_blocks) +
                f"\n\nQuestion: {state['current_query']}"
            )
        )
    ])

    answer = response.content
    if isinstance(answer, list):
        answer = " ".join(p.get("text", "") for p in answer)

    state["final_answer"] = {
        "answer": answer.strip(),
        "sources": sources,
        "policy_citations": policy_citations,
    }

    return state


# ─────────────────────────────────────────────────────────────
# Retry Router
# ─────────────────────────────────────────────────────────────
def should_retry_or_continue(state: GraphState) -> str:
    if not state["is_relevant"] and state["iteration_count"] < state["max_iterations"]:
        return "retry"
    return "continue"


# ─────────────────────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────────────────────
def build_query_graph():
    graph = StateGraph(GraphState)

    graph.add_node("query_validation", query_validation_agent)
    graph.add_node("router", router_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("rerank", rerank_node)
    graph.add_node("validate", validate_agent)
    graph.add_node("generate_answer", generate_answer_agent)

    graph.set_entry_point("query_validation")

    # Initial flow
    graph.add_edge("query_validation", "router")
    graph.add_edge("retrieval", "rerank")
    graph.add_edge("rerank", "validate")

    # Route decision
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "document": "retrieval",
            "banking": "nl2sql",
            "hybrid": "nl2sql",
        }
    )

    # Retry loop (document / hybrid only)
    graph.add_conditional_edges(
        "validate",
        should_retry_or_continue,
        {
            "retry": "retrieval",
            "continue": "generate_answer",
        }
    )

    # ✅ Separate banking vs hybrid after NL2SQL
    def after_nl2sql(state: GraphState) -> str:
        if state["route"] == "hybrid":
            return "continue_rag"
        return "end"

    graph.add_conditional_edges(
        "nl2sql",
        after_nl2sql,
        {
            "continue_rag": "retrieval",
            "end": END,
        }
    )

    graph.add_edge("generate_answer", END)

    return graph.compile()
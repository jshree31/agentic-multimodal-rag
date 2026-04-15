from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


# ─────────────────────────────────────────────────────────────────────────────
# Request Schema
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    k: int = Field(10, ge=1, le=20, description="Number of chunks to retrieve")
    chunk_type: Optional[str] = Field(
        None, description="Filter by content type: 'text', 'table', or 'image'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT QUERY RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

class DocumentQueryResponse(BaseModel):
    query: str = Field(description="User query")
    answer: str = Field(description="Answer derived from document content")
    route: str = Field(description="Route taken (should be 'document')")
    relevant_chunks: List[Dict[str, Any]] = Field(
        description="Chunks used to generate the answer"
    )

    iterations: int = Field(
        description="Number of retrieval-validation iterations"
    )

    search_type: Optional[str] = Field(
        description="Search strategy used (vector_search, fts_search, hybrid_search)"
    )

    policy_citations: str = Field(
        description="Policy citations derived from documents"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SQL / PRODUCT QUERY RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

class SQLQueryResponse(BaseModel):
    query: str = Field(description="User query")
    answer: str = Field(description="Answer derived from structured database query")
    route: str = Field(description="Route taken (should be 'banking' or 'hybrid')")
    iterations: int = Field(
        description="Number of retrieval-validation iterations"
    )

    database_name: str = Field(
        description="Database used to answer the query"
    )

    sql_query_executed: Optional[str] = Field(
        description="SQL query executed to retrieve the answer"
    )


# ─────────────────────────────────────────────────────────────────────────────
# UNION RESPONSE (for FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

QueryResponse = Union[
    DocumentQueryResponse,
    SQLQueryResponse,
]


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL SCHEMA (USED BY NL2SQL NODE ONLY)
# ─────────────────────────────────────────────────────────────────────────────

class AIResponse(BaseModel):
    query: str = Field(description="Original user query")
    answer: str = Field(description="Generated response")
    route: str = Field(description="Route taken (document, banking, hybrid)")
    iterations: int = Field(description="Number of retrieval-validation iterations")            
    policy_citations: str = Field(description="Policy citation text")
    page_no: str = Field(description="Page number reference")
    document_name: str = Field(description="Document or database name")
    sql_query_executed: Optional[str] = Field(
        default=None,
        description="SQL executed (only for SQL/product queries)"
    )
    sql_result: Optional[Any] = Field(
        default=None, description="SQL result data (only for SQL/product queries)"
    )
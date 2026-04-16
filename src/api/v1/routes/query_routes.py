from fastapi import APIRouter
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import run_query

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Executes an agentic query and returns
    either a DocumentQueryResponse or SQLQueryResponse
    depending on routing.
    """
    print(
        f"Received query: {request.query} | "
        f"k={request.k} | chunk_type={request.chunk_type}"
    )

    result = run_query(
        query=request.query,
        k=request.k,
        chunk_type=request.chunk_type
    )

    # ✅ Return as-is. FastAPI will validate against Union schema.
    return result

import base64
import hashlib
import json
import os
import pathlib

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import os

load_dotenv()

# ---------------------------------------------------------------------------
# Connection setup
#
# The .env connection string uses SQLAlchemy's dialect prefix
# "postgresql+psycopg://" so that LangChain can parse it.
# psycopg.connect() expects the standard "postgresql://" URI, so we strip
# the dialect marker before passing it to psycopg.
# ---------------------------------------------------------------------------
_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING", "")
_PG_DSN = _PG_CONNECTION.replace("postgresql+psycopg://", "postgresql://")
_RDBMS_CONNECTION = os.getenv("AGENTIC_RAG_DB_URL", "")
# How many chunks to embed per API call.
# Google's embedding API accepts up to 100 texts per batch.
_EMBED_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Issue 8 fix: Module-level embeddings singleton — avoids re-instantiating a
# new HTTP client on every store_chunks() / similarity_search() call.
# ---------------------------------------------------------------------------
get_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536,
)

# ---------------------------------------------------------------------------
# Issue 9 fix: Lazy connection pool — reuses existing TCP connections instead
# of opening a new one per request. Created on first use to avoid failing at
# import time when the DB is not yet available (e.g. during tests).
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Return the module-level connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def get_db_conn():
    """Return a pooled connection context manager.

    Usage:
        with get_db_conn() as conn:
            with conn.cursor() as cur: ...
    """
    return _get_pool().connection()

# ---------------------------------------------------------------------------
# Chunk listing (for preview / debugging)
# ---------------------------------------------------------------------------

def get_all_chunks(chunk_type: str | None = None, limit: int = 200) -> list[dict]:
    """Return all stored chunks, optionally filtered by type.

    Args:
        chunk_type: Optional filter — 'text', 'table', or 'image'.
        limit:      Max rows to return (default 200, safety cap).

    Returns:
        List of dicts with keys: id, content, chunk_type, page_number,
        section, source_file, element_type, image_base64, mime_type,
        position, metadata.
    """
    type_clause = "WHERE chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            id, content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata
        FROM multimodal_chunks
        {type_clause}
        ORDER BY page_number ASC NULLS LAST, id ASC
        LIMIT %(limit)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"chunk_type": chunk_type, "limit": limit})
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    return results

# --------------------------------------------------------------------------- 
# NEW: Search helpers (added after get_all_chunks)
# --------------------------------------------------------------------------- 

# --------------------------------------------------------------------------- 
# SEARCH FUNCTIONS + FTS SETUP
# --------------------------------------------------------------------------- 

def ensure_fts_setup():
    """One-time setup: add content_tsv column + GIN index for FTS."""
    sql = """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'multimodal_chunks' AND column_name = 'content_tsv'
            ) THEN
                ALTER TABLE multimodal_chunks ADD COLUMN content_tsv tsvector;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes 
                WHERE tablename = 'multimodal_chunks' AND indexname = 'multimodal_chunks_content_tsv_idx'
            ) THEN
                CREATE INDEX multimodal_chunks_content_tsv_idx 
                ON multimodal_chunks USING GIN(content_tsv);
            END IF;
        END $$;

        UPDATE multimodal_chunks 
        SET content_tsv = to_tsvector('english', COALESCE(content, ''))
        WHERE content_tsv IS NULL;
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("[DB] FTS setup completed (content_tsv column + index + backfill)")


def vector_search_db(query: str, k: int = 5, chunk_type: str | None = None) -> list[dict]:
    """Semantic vector search using pgvector."""
    sql = """
        SELECT content, chunk_type, element_type, page_number, section, 
               source_file, position, image_path
        FROM multimodal_chunks
        WHERE 1=1
    """
    params = {}

    if chunk_type:
        sql += " AND chunk_type = %(chunk_type)s"
        params["chunk_type"] = chunk_type

    sql += """
        ORDER BY embedding <=> %(query_emb)s::vector
        LIMIT %(k)s
    """

    emb = get_embeddings_model.embed_query(query)
    params["query_emb"] = "[" + ",".join(map(str, emb)) + "]"
    params["k"] = k

    with get_db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        image_base64 = None
        if img_path and os.path.exists(img_path):
            image_base64 = base64.b64encode(pathlib.Path(img_path).read_bytes()).decode()

        results.append({
            "content": row["content"],
            "metadata": {
                "chunk_type": row.get("chunk_type"),
                "element_type": row.get("element_type"),
                "page_number": row.get("page_number"),
                "section": row.get("section"),
                "source_file": row.get("source_file"),
                "position": row.get("position"),
                "image_base64": image_base64,
            },
            "retrieval_type": "vector"
        })
    return results


def fts_search_db(query: str, k: int = 5, chunk_type: str | None = None) -> list[dict]:
    """Full-text search."""
    sql = """
        SELECT content, chunk_type, element_type, page_number, section, 
               source_file, position, image_path
        FROM multimodal_chunks
        WHERE content_tsv @@ plainto_tsquery('english', %(query)s)
    """
    params = {"query": query, "k": k}

    if chunk_type:
        sql += " AND chunk_type = %(chunk_type)s"

    sql += """
        ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %(query)s)) DESC
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        image_base64 = None
        if img_path and os.path.exists(img_path):
            image_base64 = base64.b64encode(pathlib.Path(img_path).read_bytes()).decode()

        results.append({
            "content": row["content"],
            "metadata": {
                "chunk_type": row.get("chunk_type"),
                "element_type": row.get("element_type"),
                "page_number": row.get("page_number"),
                "section": row.get("section"),
                "source_file": row.get("source_file"),
                "position": row.get("position"),
                "image_base64": image_base64,
            },
            "retrieval_type": "fts"
        })
    return results


def hybrid_search_db(query: str, k: int = 5, chunk_type: str | None = None) -> list[dict]:
    """Simple RRF-based hybrid search."""
    vector_results = vector_search_db(query, k=k*2, chunk_type=chunk_type)
    fts_results = fts_search_db(query, k=k*2, chunk_type=chunk_type)

    combined = {}
    for rank, item in enumerate(vector_results):
        key = item["content"][:150]
        score = 1 / (rank + 60)
        combined[key] = (score, item)

    for rank, item in enumerate(fts_results):
        key = item["content"][:150]
        score = 1 / (rank + 60)
        if key in combined:
            combined[key] = (combined[key][0] + score, item)
        else:
            combined[key] = (score, item)

    ranked = sorted(combined.items(), key=lambda x: x[1][0], reverse=True)
    return [item for _, (_, item) in ranked[:k]]
# --------------------------------------------------------------------------- 
# NEW: LangChain SQLDatabase wrapper (for NL2SQL)
# --------------------------------------------------------------------------- 


def get_sql_database() -> SQLDatabase:
    """Return LangChain SQLDatabase connected to the PRODUCT RDBMS (agentic_rag_db)."""
    
    connection_string = os.getenv("AGENTIC_RAG_DB_URL", "").strip()

    if not connection_string:
        raise ValueError("AGENTIC_RAG_DB_URL is missing in .env file. This is required for product/NL2SQL queries.")

    # Ensure correct dialect
    if connection_string.startswith("postgresql://"):
        connection_string = connection_string.replace("postgresql://", "postgresql+psycopg://", 1)

    print(f"[get_sql_database] Connecting to: {connection_string[:80]}...")

    engine = create_engine(
        connection_string,
        connect_args={"options": "-c search_path=public"},
        echo=False,
    )

    from sqlalchemy import inspect
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    print(f"[get_sql_database] Available tables → {existing_tables}")

    # Only use tables that actually exist
    candidate_tables = ["products", "categories", "orders", "order_items"]
    include_tables = [t for t in candidate_tables if t in existing_tables]

    if not include_tables:
        print("[get_sql_database] No matching product tables found. Using all tables.")
        include_tables = None

    return SQLDatabase(
        engine,
        sample_rows_in_table_info=3,
        include_tables=include_tables,
    )
from langchain.tools import tool
import psycopg
from psycopg.rows import dict_row
from core.db import _raw_conn_str


@tool
def fts_search(query: str, k: int = 5, collection_name: str = ""):
    """
    Full-text keyword search using PostgreSQL FTS.
    Best for exact terms, names, or identifiers.
    """
    sql = """
        SELECT
            e.document AS content,
            e.cmetadata AS metadata,
            ts_rank(
                to_tsvector('english', e.document),
                plainto_tsquery('english', %(query)s)
            ) AS fts_rank
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = %(collection)s
          AND to_tsvector('english', e.document)
              @@ plainto_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC
        LIMIT %(k)s;
    """

    with psycopg.connect(_raw_conn_str, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query, "collection": collection_name, "k": k})
            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "metadata": row["metadata"],
            "retrieval_type": "fts",
        }
        for row in rows
    ]
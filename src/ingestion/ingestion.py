import os
import pathlib
import base64
import hashlib
import json
import logging
import time
from typing import List, Dict

from dotenv import load_dotenv

from src.core.db import _get_pool
from src.ingestion.docling_parser import parse_document

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300
_EMBED_BATCH_SIZE = 32
_MAX_EMBED_RETRIES = 3

# ---------------------------------------------------------------------------
# DB Connection
# ---------------------------------------------------------------------------
def get_db_conn():
    return _get_pool().connection()

# ---------------------------------------------------------------------------
# Document Registry
# ---------------------------------------------------------------------------
def upsert_document(filename: str, source_path: str) -> str:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, source_path)
                VALUES (%s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET source_path = EXCLUDED.source_path,
                        ingested_at  = now()
                RETURNING id
                """,
                (filename, source_path),
            )
            row = cur.fetchone()
        conn.commit()

    return str(row["id"])


# ---------------------------------------------------------------------------
# Embedding with Retry
# ---------------------------------------------------------------------------
def embed_with_retry(model, texts: List[str]) -> List[List[float]]:
    for attempt in range(_MAX_EMBED_RETRIES):
        try:
            return model.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Embedding failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError("Embedding failed after retries")


# ---------------------------------------------------------------------------
# Chunk Storage
# ---------------------------------------------------------------------------
def store_chunks(
    chunks: List[Dict],
    doc_id: str,
    get_embeddings_model
) -> int:

    if not chunks:
        return 0

    model = get_embeddings_model

    _DEDICATED_COLUMNS = {
        "content_type", "element_type", "section",
        "page_number", "source_file", "position", "image_base64",
    }

    rows_inserted = 0

    with get_db_conn() as conn:
        try:
            with conn.cursor() as cur:

                # ✅ Delete old chunks
                cur.execute(
                    "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
                    (doc_id,),
                )

                img_dir = pathlib.Path("data/images")
                img_dir.mkdir(parents=True, exist_ok=True)

                # 🔥 PROCESS IN BATCHES (embed + insert together)
                for i in range(0, len(chunks), _EMBED_BATCH_SIZE):

                    batch = chunks[i:i + _EMBED_BATCH_SIZE]
                    texts = [c["content"] for c in batch]

                    print(f"[DEBUG] Processing batch {i} → size {len(batch)}")

                    # ✅ Embed safely
                    #embeddings = embed_with_retry(model, texts)
                    embeddings = []
                    for text in texts:
                        emb = embed_with_retry(model, [text])  # send single item
                        if not emb or len(emb) == 0:
                            raise ValueError("Empty embedding returned")
                        embeddings.append(emb[0])

                    if len(embeddings) != len(batch):
                        raise ValueError(
                            f"Embedding mismatch: got {len(embeddings)} for {len(batch)} chunks"
                        )

                    # ✅ Insert SAME batch
                    for chunk, embedding in zip(batch, embeddings):
                        meta = chunk["metadata"].copy()

                        # ── Image handling ──
                        img_b64 = meta.get("image_base64")
                        image_path = None
                        mime_type = None

                        if img_b64:
                            image_bytes = base64.b64decode(img_b64)
                            img_hash = hashlib.sha256(image_bytes).hexdigest()
                            img_file = img_dir / f"{img_hash}.png"

                            if not img_file.exists():
                                img_file.write_bytes(image_bytes)

                            image_path = str(img_file)
                            mime_type = "image/png"

                        # ── Vector formatting ──
                        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                        # Clean metadata
                        clean_meta = {
                            k: v for k, v in meta.items()
                            if k not in _DEDICATED_COLUMNS
                        }

                        cur.execute(
                            """
                            INSERT INTO multimodal_chunks (
                                doc_id, chunk_type, element_type, content,
                                image_path, mime_type,
                                page_number, section, source_file,
                                position, embedding, metadata
                            ) VALUES (
                                %s::uuid, %s, %s, %s,
                                %s, %s,
                                %s, %s, %s,
                                %s::jsonb, %s::vector, %s::jsonb
                            )
                            """,
                            (
                                doc_id,
                                chunk["content_type"],
                                meta.get("element_type"),
                                chunk["content"],
                                image_path,
                                mime_type,
                                meta.get("page_number"),
                                meta.get("section"),
                                meta.get("source_file"),
                                json.dumps(meta.get("position")) if meta.get("position") else None,
                                embedding_str,
                                json.dumps(clean_meta),
                            ),
                        )

                        rows_inserted += 1

                    # ✅ Commit per batch (VERY IMPORTANT)
                    conn.commit()

                    print(f"[DEBUG] Inserted so far: {rows_inserted}")

        except Exception as e:
            conn.rollback()
            logger.error(f"DB insert failed: {e}")
            raise

    print(f"[FINAL] Total rows inserted: {rows_inserted}")
    return rows_inserted


# ---------------------------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------------------------
def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size]

        # Try to avoid cutting mid-sentence
        if len(chunk) == chunk_size:
            last_period = chunk.rfind(".")
            if last_period > chunk_size * 0.5:
                chunk = chunk[:last_period + 1]

        chunks.append(chunk)
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_ingestion(file_path: str, get_embeddings_model) -> dict:

    resolved = pathlib.Path(file_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    if resolved.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files supported")

    # Step 1: Register document
    doc_id = upsert_document(resolved.name, str(resolved))
    logger.info(f"doc_id={doc_id} file={file_path}")

    # Step 2: Parse
    parsed_elements = parse_document(file_path)

    print("DEBUG parsed_elements:", parsed_elements[:5])
    print("DEBUG count:", len(parsed_elements))

    # Validation
    for elem in parsed_elements:
        if "content" not in elem or "metadata" not in elem:
            raise ValueError("Invalid parsed element format")

    # Step 3: Chunking
    chunks: List[Dict] = []

    for elem in parsed_elements:
        if (
            elem["content_type"] == "text"
            and len(elem["content"]) > _TEXT_CHUNK_SIZE
        ):
            for sub in _split_text(
                elem["content"],
                _TEXT_CHUNK_SIZE,
                _TEXT_CHUNK_OVERLAP
            ):
                chunks.append({
                    "content": sub,
                    "content_type": elem["content_type"],
                    "metadata": elem["metadata"].copy(),  # FIXED
                })
        else:
            chunks.append(elem)

    logger.info(f"{len(chunks)} chunks ready")

    # Step 4: Store
    count = store_chunks(chunks, doc_id, get_embeddings_model)

    logger.info(f"Stored {count} chunks")

    return {
        "status": "success",
        "doc_id": doc_id,
        "chunks_ingested": count,
    }


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from src.core.db import get_embeddings_model

    model = get_embeddings_model

    if len(sys.argv) >= 2:
        pdf_path = pathlib.Path(sys.argv[1])
    else:
        pdf_path = pathlib.Path("data/KB_Smart_Banking.pdf")

    result = run_ingestion(str(pdf_path), model)
    
    print("\nIngestion complete:", result)
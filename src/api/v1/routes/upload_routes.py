from fastapi import APIRouter, UploadFile, File
import os

from src.ingestion.ingestion import run_ingestion
from src.core.db import get_embeddings_model

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # RUN INGESTION
    result = run_ingestion(file_path, get_embeddings_model)

    return {
        "message": "File uploaded & ingested successfully",
        "result": result
    }
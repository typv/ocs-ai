import os
import shutil
import uuid
from typing import Dict, Any
from datetime import datetime
from fastapi import UploadFile, File, HTTPException, APIRouter

from ..indexing.pdf_processor import process_and_add_pdf
from ...globals import global_resources, get_db_collection

api_router = APIRouter(
    prefix="/indexing"
)

@api_router.post("/upload-pdf")
async def upload_pdf(
        file: UploadFile = File(...),
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        detect_language: bool = True
) -> Dict[str, Any]:
    """
    API endpoint to upload a PDF file, process it, and save the vectorized chunks to ChromaDB.

    Args:
        file: PDF file to upload
        chunk_size: Size of each text chunk (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
        detect_language: Whether to detect language automatically (default: True)

    Returns:
        Processing result with details
    """
    # Validate file type
    if file.content_type != 'application/pdf':
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF is supported."
        )

    # Get resources
    collection = get_db_collection()
    TEMP_UPLOAD_DIR = global_resources.get("TEMP_UPLOAD_DIR")

    # Ensure temp directory exists
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

    # Generate unique filename to prevent collisions
    unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(TEMP_UPLOAD_DIR, unique_filename)

    try:
        # Save uploaded file to temporary location
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate source ID from filename (without extension)
        source_id = os.path.splitext(file.filename)[0]

        # Process PDF using the imported function
        chunk_ids, stats = process_and_add_pdf(
            file_path=file_path,
            collection=collection,
            source_id=source_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            detect_language=detect_language
        )

        # Prepare response
        response = {
            "status": "success",
            "message": "PDF processed successfully",
            "filename": file.filename,
            "chunks_added": len(chunk_ids),
            "total_vectors_in_db": collection.count(),
            "processing_timestamp": datetime.now().isoformat(),
            "source_id": source_id,
            "processing_stats": {
                "total_chunks": stats.get("total_chunks", 0),
                "processed_pages": stats.get("processed_pages", 0),
                "total_pages": stats.get("total_pages", 0),
                "language_distribution": stats.get("language_distribution", {}),
                "avg_words_per_page": stats.get("avg_words_per_page", 0),
                "avg_chunk_size": stats.get("avg_chunk_size", 0)
            }
        }

        # Add language-specific info if available
        if "chunks_by_language" in stats:
            response["processing_stats"]["chunks_by_language"] = stats["chunks_by_language"]

        return response

    except Exception as e:
        # Log error for debugging
        print(f"Error processing PDF {file.filename}: {str(e)}")

        # Clean up temp file if it exists
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )

    finally:
        # Always clean up temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
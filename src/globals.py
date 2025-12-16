import asyncio
import random

from typing import Dict, Any, AsyncGenerator
from fastapi import HTTPException

API_PREFIX = '/api/v1'

# --- Global Resources Dictionary ---
global_resources: Dict[str, Any] = {}

# --- Global Constants ---
SINGLETON_THREAD_ID = "addeadf1-8411-4e12-94ce-3e783c218c3a"

def get_db_collection():
    """Retrieves the ChromaDB collection instance."""
    collection = global_resources.get("document_collection")
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not available or collection not initialized.")
    return collection


async def stream_text_naturally(text: str, min_chunk_size: int = 5, max_chunk_size: int = 15) -> AsyncGenerator[
    str, None]:
    """
    Streams a final text answer in chunks (sequences of characters)
    with randomized chunk sizes and punctuation pauses to mimic LLM output.
    """
    BASE_MIN_DELAY = 0.01
    BASE_MAX_DELAY = 0.05

    PUNCTUATION_DELAY = 0.2
    PUNCTUATION_MARKS = ['.', '!', '?', ';', '\n']

    i = 0
    while i < len(text):
        remaining = len(text) - i
        chunk_size = random.randint(min_chunk_size, max_chunk_size)
        current_chunk_size = min(chunk_size, remaining)
        chunk = text[i: i + current_chunk_size]
        delay = random.uniform(BASE_MIN_DELAY, BASE_MAX_DELAY)
        if chunk[-1] in PUNCTUATION_MARKS:
            delay += PUNCTUATION_DELAY
        elif '\n' in chunk:
            delay += PUNCTUATION_DELAY / 2

        yield chunk
        i += current_chunk_size

        await asyncio.sleep(delay)
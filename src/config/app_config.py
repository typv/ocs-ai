import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from .chroma_config import ChromaSettings
from .gemini_config import GeminiSettings
from .google_search_config import GoogleSearchSettings
from .rag_config import RAGSettings
from functools import lru_cache
from dotenv import load_dotenv
from pathlib import Path

DOTENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

load_dotenv(dotenv_path=DOTENV_PATH, override=True)

class Settings(BaseSettings):
    CHROMA: ChromaSettings = ChromaSettings()
    GEMINI: GeminiSettings = GeminiSettings()
    GOOGLE_SEARCH: GoogleSearchSettings = GoogleSearchSettings()
    RAG: RAGSettings = RAGSettings()

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        case_sensitive=True,
        extra='ignore'
    )

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
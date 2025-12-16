import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from .chroma_config import ChromaSettings
from .gemini_config import GeminiSettings
from .google_search_config import GoogleSearchSettings
from .rag_config import RAGSettings

DOTENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')

class Settings(BaseSettings):
    CHROMA: ChromaSettings = ChromaSettings()
    GEMINI: GeminiSettings = GeminiSettings()
    GOOGLE_SEARCH: GoogleSearchSettings = GoogleSearchSettings()
    RAG: RAGSettings = RAGSettings()

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        case_sensitive=True
    )


settings = Settings()
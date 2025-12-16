from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGSettings(BaseSettings):
    RETRIEVAL_K: int = 20
    CONTEXT_TOP_N: int = 8

    model_config = SettingsConfigDict(
        env_prefix='RAG_',
        case_sensitive=True
    )
from pydantic_settings import BaseSettings, SettingsConfigDict

class ChromaSettings(BaseSettings):
    HOST: str
    PORT: int
    DEFAULT_COLLECTION_NAME: str = "default_collection"

    model_config = SettingsConfigDict(
        env_prefix='CHROMA_',
        case_sensitive=True
    )
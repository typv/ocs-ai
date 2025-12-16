from pydantic_settings import BaseSettings, SettingsConfigDict


class GoogleSearchSettings(BaseSettings):
    API_KEY: str
    CX_ID: str
    ENABLED: bool = False

    model_config = SettingsConfigDict(
        env_prefix='GOOGLE_SEARCH_',
        case_sensitive=True
    )
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    MODEL: str = "gemini-2.5-flash"
    API_KEY: str

    model_config = SettingsConfigDict(
        env_prefix='GEMINI_',
        case_sensitive=True
    )
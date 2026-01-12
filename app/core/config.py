from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "mp-rag-backend"
    app_version: str = "0.1.0"
    vector_dim: int = 384
    max_context_chars: int = 2000
    model_name: str = "stub-llm"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    temperature: float = 0.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()

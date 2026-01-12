from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "mp-rag-backend"
    app_version: str = "0.1.0"
    vector_dim: int = 384
    max_context_chars: int = 2000
    temperature: float = 0.0
    vitoria_model: str = "vitoria-ai"
    vitoria_api_key: str | None = None
    vitoria_base_url: str = "http://localhost:9000"
    # Vector/Document stores
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_vectors"
    mongo_uri: str | None = None
    mongo_db: str = "rag"
    mongo_collection: str = "documents"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


def get_settings() -> Settings:
    return Settings()

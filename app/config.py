from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    During local development these values can come from a `.env` file.
    In production they should come from the hosting platform's secret manager.

    This keeps secrets and deploy-time choices out of the source code.
    """

    app_name: str = "SHL Assessment Recommender"
    app_env: str = "local"

    catalog_url: str = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    catalog_path: str = "app/data/shl_product_catalog.json"
    faiss_index_path: str = "app/data/shl_catalog.faiss"
    metadata_path: str = "app/data/shl_catalog_metadata.json"
    model_cache_dir: str = "app/data/model_cache"

    gemini_api_key: str = Field(default="", repr=False)
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 512

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_recommendations: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )


@lru_cache
def get_settings() -> Settings:
    """Cache settings so they are created once per process.

    Without caching, settings would be re-read repeatedly.
    """
    return Settings()


settings = get_settings()

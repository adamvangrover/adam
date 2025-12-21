from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration management using Pydantic Settings.
    Loads from environment variables and .env file.
    """
    # System
    app_name: str = "Adam v23.5"
    debug: bool = False
    log_level: str = "INFO"
    environment: str = "production"

    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None

    # Database / Infrastructure
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    postgres_db: Optional[str] = None
    database_url: Optional[str] = None

    redis_url: str = "redis://redis:6379/0"
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[str] = None

    # Paths
    config_path: str = "config/"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

# Global settings instance
settings = Settings()

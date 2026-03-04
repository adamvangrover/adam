import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # For debugging and testing purposes
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    class Config:
        env_file = ".env"

settings = Settings()

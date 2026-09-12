from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONE_", env_file=".env", extra="ignore")
    env: str = "development"
    database_url: str = "sqlite:///./one.db"
    object_store_path: Path = Path("./data/objects")
    clip_encryption_key_b64: str | None = None
    bootstrap_secret: str = "change-me-in-production"
    session_ttl_minutes: int = 60
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None
    lm_studio_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_model: str = "qwen3.6-35b-a3b"
    lm_studio_api_key: str | None = Field(default=None, validation_alias=AliasChoices("ONE_LM_STUDIO_API_KEY", "ONE_LLM_API_KEY", "LLM_API_KEY"))
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:4173,http://localhost:4174,http://127.0.0.1:5173,http://127.0.0.1:4173,http://127.0.0.1:4174"

    @property
    def cors_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def sqlite_path(self) -> str | None:
        if self.database_url.startswith("sqlite:///"):
            return self.database_url.removeprefix("sqlite:///")
        return None

    @property
    def database_backend(self) -> str:
        """Return the supported storage backend selected by ONE_DATABASE_URL."""
        scheme = self.database_url.split(":", 1)[0].lower()
        if scheme == "sqlite":
            return "sqlite"
        if scheme in {"postgres", "postgresql"}:
            return "postgresql"
        return "unsupported"


@lru_cache
def get_settings() -> Settings:
    return Settings()

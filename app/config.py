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
    # OpenAI-compatible provider overrides. The legacy LM Studio fields above
    # remain the defaults so existing .env files and Settings(...) callers are
    # unchanged. Keys are only used in-memory and are never included in logs or
    # response models.
    llm_enabled: bool = True
    llm_provider: str | None = None
    llm_base_url: str | None = Field(default=None, validation_alias=AliasChoices("ONE_LLM_BASE_URL", "LLM_BASE_URL"))
    llm_model: str | None = Field(default=None, validation_alias=AliasChoices("ONE_LLM_MODEL", "LLM_MODEL"))
    llm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("ONE_LLM_API_KEY", "LLM_API_KEY"))
    llm_timeout_seconds: float = Field(default=15.0, gt=0, validation_alias=AliasChoices("ONE_LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS"))
    geometry_service_url: str = Field(default="http://host.docker.internal:8090", validation_alias=AliasChoices("ONE_GEOMETRY_SERVICE_URL", "GEOMETRY_SERVICE_URL"))
    geometry_timeout_seconds: float = Field(default=45.0, gt=0, le=120, validation_alias=AliasChoices("ONE_GEOMETRY_TIMEOUT_SECONDS", "GEOMETRY_TIMEOUT_SECONDS"))
    geometry_require_gpu: bool = Field(default=True, validation_alias=AliasChoices("ONE_GEOMETRY_REQUIRE_GPU", "GEOMETRY_REQUIRE_GPU"))
    geometry_min_confidence: float = Field(default=0.65, ge=0, le=1, validation_alias=AliasChoices("ONE_GEOMETRY_MIN_CONFIDENCE", "GEOMETRY_MIN_CONFIDENCE"))
    geometry_max_reprojection_error_px: float = Field(default=48.0, gt=0, le=1000, validation_alias=AliasChoices("ONE_GEOMETRY_MAX_REPROJECTION_ERROR_PX", "GEOMETRY_MAX_REPROJECTION_ERROR_PX"))
    geometry_min_homography_inlier_ratio: float = Field(default=0.5, ge=0, le=1, validation_alias=AliasChoices("ONE_GEOMETRY_MIN_HOMOGRAPHY_INLIER_RATIO", "GEOMETRY_MIN_HOMOGRAPHY_INLIER_RATIO"))
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:4173,http://localhost:4174,http://localhost:4175,http://127.0.0.1:5173,http://127.0.0.1:4173,http://127.0.0.1:4174,http://127.0.0.1:4175"

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

    @property
    def effective_llm_base_url(self) -> str:
        return (self.llm_base_url or self.lm_studio_url).rstrip("/")

    @property
    def effective_llm_model(self) -> str:
        return self.llm_model or self.lm_studio_model

    @property
    def effective_llm_api_key(self) -> str | None:
        return self.llm_api_key or self.lm_studio_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()

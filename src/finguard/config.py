"""Application settings loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden with a FINGUARD_* variable."""

    model_config = SettingsConfigDict(env_prefix="FINGUARD_", env_file=".env", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"
    log_json: bool = False
    model_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    policy_path: Path = PROJECT_ROOT / "configs" / "policies" / "committee_v1.yaml"
    api_url: str = "http://localhost:8000"
    random_seed: int = 42

    def resolve(self, path: Path) -> Path:
        """Resolve a possibly relative path against the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings()

from pathlib import Path
from typing import List

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application configuration loaded from environment/.env.

    Uses pydantic-settings (v2) to read values and provide defaults.
    """

    # Paths (relative to project root by default)
    DATA_DIR: str = "./data"
    PERSIST_BASE_DIR: str = "./storage"

    # Providers
    GOOGLE_API_KEY: str
    LLM_MODEL: str = "gemini-1.5-flash"
    EMBED_MODEL: str = "gemini-embedding-001"

    # Runtime defaults (can be changed via /config)
    RAG_MODE: str = "auto"  # auto|rag_only|llm_only
    RAG_TOP_K: int = 4
    RAG_CUTOFF: float = 0.3

    # System prompt default
    DEFAULT_SYSTEM_PROMPT: str = (
        "Ти дружній україномовний асистент. Відповідай лаконічно, зрозуміло, \n"
        "дотримуйся ввічливого тону, за потреби наводь списки та приклади."
    )
    SYSTEM_PROMPT: str | None = None

    # CORS
    CORS_ALLOW_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def base_dir(self) -> Path:
        # project root is two levels up from this file
        return Path(__file__).resolve().parents[2]

    @property
    def data_path(self) -> Path:
        p = Path(self.DATA_DIR)
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def persist_base_path(self) -> Path:
        p = Path(self.PERSIST_BASE_DIR)
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def system_prompt_effective(self) -> str:
        return (self.SYSTEM_PROMPT or self.DEFAULT_SYSTEM_PROMPT).strip()


class RuntimeConfig(BaseModel):
    rag_mode: str
    rag_top_k: int
    rag_cutoff: float
    system_prompt: str

    @field_validator("rag_mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        vv = (v or "").strip().lower()
        if vv not in ("auto", "rag_only", "llm_only"):
            raise ValueError("rag_mode must be one of: auto|rag_only|llm_only")
        return vv

    @classmethod
    def from_settings(cls, s: AppSettings) -> "RuntimeConfig":
        return cls(
            rag_mode=s.RAG_MODE,
            rag_top_k=s.RAG_TOP_K,
            rag_cutoff=s.RAG_CUTOFF,
            system_prompt=s.system_prompt_effective,
        )


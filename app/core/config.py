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

    # Indexing
    INDEX_CHUNK_SIZE: int = 126
    INDEX_CHUNK_OVERLAP: int = 20

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

    @field_validator("INDEX_CHUNK_SIZE", "INDEX_CHUNK_OVERLAP")
    @classmethod
    def _validate_non_negative(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError("INDEX_CHUNK_SIZE/INDEX_CHUNK_OVERLAP must be >= 0")
        return int(v)

    @field_validator("INDEX_CHUNK_OVERLAP")
    @classmethod
    def _validate_overlap_bound(cls, v: int, info):
        # Cross-field check is finalized in model validator below; keep basic int cast here
        return int(v)

    # Ensure overlap < size after all fields are populated
    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        if self.INDEX_CHUNK_OVERLAP >= self.INDEX_CHUNK_SIZE:
            raise ValueError(
                "INDEX_CHUNK_OVERLAP must be smaller than INDEX_CHUNK_SIZE"
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
    llm_model: str
    embed_model: str
    index_chunk_size: int
    index_chunk_overlap: int

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
            llm_model=s.LLM_MODEL,
            embed_model=s.EMBED_MODEL,
            index_chunk_size=s.INDEX_CHUNK_SIZE,
            index_chunk_overlap=s.INDEX_CHUNK_OVERLAP,
        )

    @field_validator("index_chunk_size", "index_chunk_overlap")
    @classmethod
    def _validate_non_negative(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError("chunk size/overlap must be >= 0")
        return int(v)

    @field_validator("index_chunk_overlap")
    @classmethod
    def _validate_overlap_bound(cls, v: int, info):
        return int(v)

    def model_post_init(self, __context) -> None:
        if self.index_chunk_overlap >= self.index_chunk_size:
            raise ValueError("index_chunk_overlap must be smaller than index_chunk_size")

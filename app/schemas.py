from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class ApplyIndexRequest(BaseModel):
    index_id: str


class ConfigUpdate(BaseModel):
    rag_mode: Optional[str] = None  # auto|rag_only|llm_only
    rag_top_k: Optional[int] = None
    rag_cutoff: Optional[float] = None
    system_prompt: Optional[str] = None


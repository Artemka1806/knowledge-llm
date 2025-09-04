from __future__ import annotations

import re
from typing import Any, Dict, List

from fastapi import HTTPException
from llama_index.core import Settings
from llama_index.core.postprocessor import SimilarityPostprocessor

from app.core.config import RuntimeConfig


def make_query_engine(index, runtime: RuntimeConfig):
    return index.as_query_engine(
        similarity_top_k=runtime.rag_top_k,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=runtime.rag_cutoff)],
        streaming=False,
    )


def should_fallback_to_llm(question: str, index, runtime: RuntimeConfig) -> bool:
    try:
        mode = (runtime.rag_mode or "auto").strip().lower()
        if mode == "rag_only":
            return False
        if mode == "llm_only":
            return True

        q = (question or "").strip().lower()
        if re.search(r"\b(привіт|добрий день|вітаю|hello|hi|hey)\b", q):
            return True

        retriever = index.as_retriever(similarity_top_k=runtime.rag_top_k)
        nodes = retriever.retrieve(question)
        if not nodes:
            return True
        best = max((n.score or 0.0) for n in nodes)
        return best < runtime.rag_cutoff
    except Exception:
        return True


def build_prompt_from_history(system_prompt: str, history: List[Dict[str, Any]], message: str) -> str:
    parts = [f"System: {system_prompt}"]
    for item in history or []:
        role = (item.get("role") or "user").lower()
        content = item.get("content") or ""
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append(f"User: {message}")
    parts.append("Assistant:")
    return "\n".join(parts)


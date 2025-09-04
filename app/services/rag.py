from __future__ import annotations

import asyncio
import re
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from llama_index.core import Settings
from llama_index.core.postprocessor import SimilarityPostprocessor

from app.core.config import RuntimeConfig


logger = logging.getLogger("rag")


def make_query_engine(index, runtime: RuntimeConfig):
    return index.as_query_engine(
        similarity_top_k=runtime.rag_top_k,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=runtime.rag_cutoff)],
        streaming=False,
    )


def _merge_extra(ctx: Optional[Dict[str, Any]], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(extra or {})
    if ctx:
        # do not override explicit extra by ctx collision
        for k, v in ctx.items():
            if k not in out:
                out[k] = v
    return out


async def should_fallback_to_llm(
    question: str,
    index,
    runtime: RuntimeConfig,
    ctx: Optional[Dict[str, Any]] = None,
) -> bool:
    try:
        mode = (runtime.rag_mode or "auto").strip().lower()
        if mode == "rag_only":
            logger.info(
                "decision:force_rag",
                extra=_merge_extra(ctx, {"reason": "rag_only", "mode": mode}),
            )
            return False
        if mode == "llm_only":
            logger.info(
                "decision:force_llm",
                extra=_merge_extra(ctx, {"reason": "llm_only", "mode": mode}),
            )
            return True

        q = (question or "").strip().lower()
        if re.search(r"\b(привіт|добрий день|вітаю|hello|hi|hey)\b", q):
            return True

        retriever = index.as_retriever(similarity_top_k=runtime.rag_top_k)
        t0 = time.perf_counter()
        nodes = None
        # Prefer async retrieval if supported
        aretrieve = getattr(retriever, "aretrieve", None)
        if callable(aretrieve):
            nodes = await aretrieve(question)
        else:
            # Offload sync retrieval to a thread to avoid blocking
            nodes = await asyncio.to_thread(retriever.retrieve, question)
        t_ms = int((time.perf_counter() - t0) * 1000)

        if not nodes:
            logger.info(
                "retrieval:empty",
                extra=_merge_extra(
                    ctx,
                    {
                        "top_k": runtime.rag_top_k,
                        "cutoff": runtime.rag_cutoff,
                        "duration_ms": t_ms,
                    },
                ),
            )
            return True
        scores = [float(getattr(n, "score", 0.0) or 0.0) for n in nodes]
        best = max(scores) if scores else 0.0

        # Stats for debugging score scale
        stats = {
            "count": len(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "avg": (sum(scores) / len(scores)) if scores else None,
        }

        # Basic scale sanity warning
        if any(s < 0.0 or s > 1.0 for s in scores):
            logger.warning(
                "scores:out_of_01_range",
                extra=_merge_extra(
                    ctx,
                    {
                        "min": stats["min"],
                        "max": stats["max"],
                        "avg": stats["avg"],
                        "cutoff": runtime.rag_cutoff,
                    },
                ),
            )

        # Log top few scores (avoid relying on node internals)
        top_preview = sorted(scores, reverse=True)[: min(5, len(scores))]
        logger.info(
            "retrieval:stats",
            extra=_merge_extra(
                ctx,
                {
                    "duration_ms": t_ms,
                    "top_k": runtime.rag_top_k,
                    "cutoff": runtime.rag_cutoff,
                    "best": best,
                    "min": stats["min"],
                    "max": stats["max"],
                    "avg": stats["avg"],
                    "top_scores": top_preview,
                },
            ),
        )

        decision = best < runtime.rag_cutoff
        logger.info(
            "decision:fallback_llm" if decision else "decision:use_rag",
            extra=_merge_extra(
                ctx,
                {
                    "best": best,
                    "cutoff": runtime.rag_cutoff,
                    "mode": mode,
                },
            ),
        )
        return decision
    except Exception:
        logger.exception("retrieval:error", extra=_merge_extra(ctx, {}))
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

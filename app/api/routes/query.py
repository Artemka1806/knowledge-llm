import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from llama_index.core import Settings

from app.core.config import RuntimeConfig
from app.services.rag import make_query_engine, should_fallback_to_llm
from app.schemas import QueryRequest


router = APIRouter()
log = logging.getLogger("rag")


@router.post("/query")
async def query_endpoint(request: Request, payload: QueryRequest):
    index = request.app.state.index
    runtime: RuntimeConfig = request.app.state.runtime
    index_id = request.app.state.active_index_id
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    ctx = {"request_id": request_id, "index_id": index_id}
    if index is None:
        raise HTTPException(status_code=503, detail="Індекс ще не готовий")

    if await should_fallback_to_llm(payload.query, index, runtime, ctx):
        llm = Settings.llm
        try:
            t0 = time.perf_counter()
            acomplete = getattr(llm, "acomplete", None)
            if callable(acomplete):
                comp = await acomplete(f"{runtime.system_prompt}\nПитання: {payload.query}")
            else:
                comp = await asyncio.to_thread(llm.complete, f"{runtime.system_prompt}\nПитання: {payload.query}")
            log.info(
                "llm:complete",
                extra={
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                    "mode": runtime.rag_mode,
                    **ctx,
                },
            )
            return {"response": str(comp)}
        except Exception as e:
            qe = make_query_engine(index, runtime)
            try:
                t0 = time.perf_counter()
                aquery = getattr(qe, "aquery", None)
                if callable(aquery):
                    resp = await aquery(payload.query)
                else:
                    resp = await asyncio.to_thread(qe.query, payload.query)
                log.info(
                    "rag:query_after_llm_fail",
                    extra={
                        "duration_ms": int((time.perf_counter() - t0) * 1000),
                        "top_k": runtime.rag_top_k,
                        "cutoff": runtime.rag_cutoff,
                        **ctx,
                    },
                )
            except Exception as inner:
                raise HTTPException(status_code=500, detail=f"Помилка запиту: {inner}")
            return {"response": str(resp), "note": f"LLM fallback failed: {e}"}
    else:
        qe = make_query_engine(index, runtime)
        try:
            t0 = time.perf_counter()
            aquery = getattr(qe, "aquery", None)
            if callable(aquery):
                resp = await aquery(payload.query)
            else:
                resp = await asyncio.to_thread(qe.query, payload.query)
            log.info(
                "rag:query",
                extra={
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                    "top_k": runtime.rag_top_k,
                    "cutoff": runtime.rag_cutoff,
                    **ctx,
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Помилка запиту: {e}")
        return {"response": str(resp)}

import asyncio

from fastapi import APIRouter, HTTPException, Request
from llama_index.core import Settings

from app.core.config import RuntimeConfig
from app.services.rag import make_query_engine, should_fallback_to_llm
from app.schemas import QueryRequest


router = APIRouter()


@router.post("/query")
async def query_endpoint(request: Request, payload: QueryRequest):
    index = request.app.state.index
    runtime: RuntimeConfig = request.app.state.runtime
    if index is None:
        raise HTTPException(status_code=503, detail="Індекс ще не готовий")

    if await should_fallback_to_llm(payload.query, index, runtime):
        llm = Settings.llm
        try:
            acomplete = getattr(llm, "acomplete", None)
            if callable(acomplete):
                comp = await acomplete(f"{runtime.system_prompt}\nПитання: {payload.query}")
            else:
                comp = await asyncio.to_thread(llm.complete, f"{runtime.system_prompt}\nПитання: {payload.query}")
            return {"response": str(comp)}
        except Exception as e:
            qe = make_query_engine(index, runtime)
            try:
                aquery = getattr(qe, "aquery", None)
                if callable(aquery):
                    resp = await aquery(payload.query)
                else:
                    resp = await asyncio.to_thread(qe.query, payload.query)
            except Exception as inner:
                raise HTTPException(status_code=500, detail=f"Помилка запиту: {inner}")
            return {"response": str(resp), "note": f"LLM fallback failed: {e}"}
    else:
        qe = make_query_engine(index, runtime)
        try:
            aquery = getattr(qe, "aquery", None)
            if callable(aquery):
                resp = await aquery(payload.query)
            else:
                resp = await asyncio.to_thread(qe.query, payload.query)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Помилка запиту: {e}")
        return {"response": str(resp)}

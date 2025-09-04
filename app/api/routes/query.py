from fastapi import APIRouter, HTTPException, Request
from llama_index.core import Settings

from app.core.config import RuntimeConfig
from app.services.rag import make_query_engine, should_fallback_to_llm
from app.schemas import QueryRequest


router = APIRouter()


@router.post("/query")
def query_endpoint(request: Request, payload: QueryRequest):
    index = request.app.state.index
    runtime: RuntimeConfig = request.app.state.runtime
    if index is None:
        raise HTTPException(status_code=503, detail="Індекс ще не готовий")

    if should_fallback_to_llm(payload.query, index, runtime):
        llm = Settings.llm
        try:
            comp = llm.complete(f"{runtime.system_prompt}\nПитання: {payload.query}")
            return {"response": str(comp)}
        except Exception as e:
            qe = make_query_engine(index, runtime)
            resp = qe.query(payload.query)
            return {"response": str(resp), "note": f"LLM fallback failed: {e}"}
    else:
        qe = make_query_engine(index, runtime)
        resp = qe.query(payload.query)
        return {"response": str(resp)}


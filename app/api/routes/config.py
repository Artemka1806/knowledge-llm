from fastapi import APIRouter, HTTPException, Request

from app.core.config import RuntimeConfig
from app.core.llm import apply_text_qa_template
from app.schemas import ConfigUpdate


router = APIRouter()


@router.get("/config")
def get_config(request: Request):
    runtime: RuntimeConfig = request.app.state.runtime
    return {
        "rag_mode": runtime.rag_mode,
        "rag_top_k": runtime.rag_top_k,
        "rag_cutoff": runtime.rag_cutoff,
        "system_prompt": runtime.system_prompt,
    }


@router.post("/config")
def set_config(request: Request, update: ConfigUpdate):
    runtime: RuntimeConfig = request.app.state.runtime

    if update.rag_mode is not None:
        mode = (update.rag_mode or "").strip().lower()
        if mode not in ("auto", "rag_only", "llm_only"):
            raise HTTPException(status_code=400, detail="Невірний rag_mode (auto|rag_only|llm_only)")
        runtime.rag_mode = mode
    if update.rag_top_k is not None:
        try:
            v = int(update.rag_top_k)
            if v < 1 or v > 50:
                raise ValueError
            runtime.rag_top_k = v
        except Exception:
            raise HTTPException(status_code=400, detail="rag_top_k має бути цілим 1..50")
    if update.rag_cutoff is not None:
        try:
            v = float(update.rag_cutoff)
            if not (0.0 <= v <= 1.0):
                raise ValueError
            runtime.rag_cutoff = v
        except Exception:
            raise HTTPException(status_code=400, detail="rag_cutoff має бути числом 0.0..1.0")
    if update.system_prompt is not None:
        runtime.system_prompt = str(update.system_prompt or runtime.system_prompt)
        try:
            apply_text_qa_template(runtime.system_prompt)
        except Exception:
            pass
    return get_config(request)


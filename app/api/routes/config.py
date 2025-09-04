from fastapi import APIRouter, HTTPException, Request

from app.core.config import AppSettings, RuntimeConfig
from app.core.llm import apply_text_qa_template, configure_llm
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
        "llm_model": runtime.llm_model,
        "embed_model": runtime.embed_model,
        "index_chunk_size": runtime.index_chunk_size,
        "index_chunk_overlap": runtime.index_chunk_overlap,
    }


@router.post("/config")
def set_config(request: Request, update: ConfigUpdate):
    settings: AppSettings = request.app.state.settings
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
            runtime.rag_cutoff = v
        except Exception:
            raise HTTPException(status_code=400, detail="rag_cutoff має бути числом")
    if update.system_prompt is not None:
        runtime.system_prompt = str(update.system_prompt or runtime.system_prompt)
        try:
            apply_text_qa_template(runtime.system_prompt)
        except Exception:
            pass
    if hasattr(update, "llm_model") and update.llm_model is not None:
        runtime.llm_model = str(update.llm_model).strip()
        if not runtime.llm_model:
            raise HTTPException(status_code=400, detail="llm_model не може бути порожнім")
        configure_llm(settings, runtime)
    if hasattr(update, "embed_model") and update.embed_model is not None:
        runtime.embed_model = str(update.embed_model).strip()
        if not runtime.embed_model:
            raise HTTPException(status_code=400, detail="embed_model не може бути порожнім")
        configure_llm(settings, runtime)
    if hasattr(update, "index_chunk_size") and update.index_chunk_size is not None:
        try:
            v = int(update.index_chunk_size)
            if v < 1:
                raise ValueError
            runtime.index_chunk_size = v
        except Exception:
            raise HTTPException(status_code=400, detail="index_chunk_size має бути цілим ≥ 1")
        # keep constraint
        if runtime.index_chunk_overlap >= runtime.index_chunk_size:
            raise HTTPException(status_code=400, detail="index_chunk_overlap має бути < index_chunk_size")
    if hasattr(update, "index_chunk_overlap") and update.index_chunk_overlap is not None:
        try:
            v = int(update.index_chunk_overlap)
            if v < 0:
                raise ValueError
            runtime.index_chunk_overlap = v
        except Exception:
            raise HTTPException(status_code=400, detail="index_chunk_overlap має бути цілим ≥ 0")
        if runtime.index_chunk_overlap >= runtime.index_chunk_size:
            raise HTTPException(status_code=400, detail="index_chunk_overlap має бути < index_chunk_size")
    return get_config(request)

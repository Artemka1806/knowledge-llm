from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.core.config import AppSettings, RuntimeConfig
from app.core.indexing import build_full_index, load_index_by_id, list_index_dirs
from app.core.llm import configure_llm
from app.schemas import ApplyIndexRequest


router = APIRouter()


@router.post("/refresh")
def refresh_index(request: Request):
    settings: AppSettings = request.app.state.settings
    runtime: RuntimeConfig = request.app.state.runtime

    # Full rebuild and activate
    index_id, idx = build_full_index(settings.data_path, settings.persist_base_path)
    request.app.state.index = idx
    request.app.state.active_index_id = index_id
    return {"status": "Переіндексовано та активовано", "index_id": index_id}


@router.post("/index/build")
def build_index_all_files(request: Request):
    settings: AppSettings = request.app.state.settings
    index_id, _ = build_full_index(settings.data_path, settings.persist_base_path)
    return {"status": "Індекс збудовано", "index_id": index_id}


@router.post("/index/apply")
def apply_index(request: Request, req: ApplyIndexRequest):
    settings: AppSettings = request.app.state.settings
    runtime: RuntimeConfig = request.app.state.runtime
    try:
        configure_llm(settings, runtime)
        idx = load_index_by_id(settings.persist_base_path, req.index_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Індекс не знайдено")
    request.app.state.active_index_id = req.index_id
    request.app.state.index = idx
    return {"status": "Індекс активовано", "index_id": req.index_id}


@router.get("/index/list")
def list_indexes(request: Request):
    settings: AppSettings = request.app.state.settings
    items = []
    for idx_id, path in list_index_dirs(settings.persist_base_path):
        items.append({
            "id": idx_id,
            "mtime": int(path.stat().st_mtime),
        })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"active": request.app.state.active_index_id, "items": items}


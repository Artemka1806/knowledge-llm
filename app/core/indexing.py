from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from app.core.config import AppSettings, RuntimeConfig


def list_index_dirs(base_dir: Path) -> List[Tuple[str, Path]]:
    if not base_dir.exists():
        return []
    items: List[Tuple[str, Path]] = []
    for child in base_dir.iterdir():
        if child.is_dir() and (child / "docstore.json").exists():
            items.append((child.name, child))
    return items


def latest_index_dir(base_dir: Path) -> Optional[Tuple[str, Path]]:
    items = list_index_dirs(base_dir)
    if not items:
        return None
    items.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    return items[0]


def load_index_by_id(persist_base: Path, index_id: str) -> VectorStoreIndex:
    storage_dir = persist_base / index_id
    if not storage_dir.exists():
        raise FileNotFoundError(f"Index '{index_id}' not found")
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    return load_index_from_storage(storage_context)


def build_full_index(settings: AppSettings, runtime: RuntimeConfig) -> tuple[str, VectorStoreIndex]:
    docs = SimpleDirectoryReader(
        str(settings.data_path),
        required_exts=[".txt", ".md"],
        recursive=True,
    ).load_data()
    # Ensure chunk_overlap < chunk_size (validated in RuntimeConfig)
    idx = VectorStoreIndex.from_documents(
        docs,
        transformations=[
            SentenceSplitter(
                chunk_size=runtime.index_chunk_size,
                chunk_overlap=runtime.index_chunk_overlap,
            )
        ],
    )
    index_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    persist_dir = settings.persist_base_path / index_id
    persist_dir.mkdir(parents=True, exist_ok=True)
    idx.storage_context.persist(persist_dir=str(persist_dir))
    return index_id, idx


def has_supported_files(data_dir: Path, exts: Iterable[str] = (".txt", ".md")) -> bool:
    if not data_dir.exists():
        return False
    exts_l = {e.lower() for e in exts}
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_l:
            return True
    return False

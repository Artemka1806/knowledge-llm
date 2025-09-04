import os
from pathlib import Path
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Tuple
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Settings,
    StorageContext, load_index_from_storage, PromptTemplate
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding

from google import genai

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_DIR = "./data"
PERSIST_BASE_DIR = "./storage"  # indexes will be stored under storage/{index_id}

# Runtime-configurable settings (can be changed via /config)
RAG_MODE = (os.getenv("RAG_MODE", "auto") or "auto").strip().lower()  # auto|rag_only|llm_only
try:
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
except Exception:
    RAG_TOP_K = 4
try:
    RAG_CUTOFF = float(os.getenv("RAG_CUTOFF", "0.3"))
except Exception:
    RAG_CUTOFF = 0.3

# System prompt runtime value
DEFAULT_SYSTEM_PROMPT = (
    "Ти дружній україномовний асистент. Відповідай лаконічно, зрозуміло, \n"
    "дотримуйся ввічливого тону, за потреби наводь списки та приклади."
)
SYSTEM_PROMPT_VALUE = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

app = FastAPI()
index = None
active_index_id = None

class QueryRequest(BaseModel):
    query: str

class ApplyIndexRequest(BaseModel):
    index_id: str


class ConfigUpdate(BaseModel):
    rag_mode: Optional[str] = None  # auto|rag_only|llm_only
    rag_top_k: Optional[int] = None
    rag_cutoff: Optional[float] = None
    system_prompt: Optional[str] = None

def _list_index_dirs() -> List[Tuple[str, Path]]:
    base = Path(PERSIST_BASE_DIR)
    if not base.exists():
        return []
    items = []
    for child in base.iterdir():
        if child.is_dir():
            # Heuristic: a valid index dir should contain docstore.json
            if (child / "docstore.json").exists():
                items.append((child.name, child))
    return items


def _latest_index_dir() -> Optional[Tuple[str, Path]]:
    items = _list_index_dirs()
    if not items:
        return None
    # Sort by mtime, descending
    items.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    return items[0]


def _load_index_by_id(index_id: str):
    storage_dir = Path(PERSIST_BASE_DIR) / index_id
    if not storage_dir.exists():
        raise FileNotFoundError(f"Index '{index_id}' not found")
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    idx = load_index_from_storage(storage_context)
    return idx


def _build_full_index() -> tuple[str, VectorStoreIndex]:
    # Configure LLM once per build
    _configure_llm()
    # Restrict to text-like files to avoid noisy HTML/CSS/JS code
    docs = SimpleDirectoryReader(
        DATA_DIR,
        required_exts=[".txt", ".md"],
        recursive=True,
    ).load_data()
    idx = VectorStoreIndex.from_documents(
        docs, transformations=[SentenceSplitter(chunk_size=512)]
    )
    # Persist under a new versioned directory
    from datetime import datetime
    index_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    persist_dir = Path(PERSIST_BASE_DIR) / index_id
    persist_dir.mkdir(parents=True, exist_ok=True)
    idx.storage_context.persist(persist_dir=str(persist_dir))
    return index_id, idx

@app.on_event("startup")
def startup_event():
    # On first run: if any index exists, take the latest; otherwise build and take it
    global index, active_index_id
    latest = _latest_index_dir()
    if latest is None:
        index_id, idx = _build_full_index()
        active_index_id = index_id
        index = idx
    else:
        active_index_id, path = latest
        # Ensure LLM configured for runtime
        _configure_llm()
        index = _load_index_by_id(active_index_id)

@app.post("/query")
def query_endpoint(request: QueryRequest):
    global index
    if index is None:
        raise HTTPException(status_code=503, detail="Індекс ще не готовий")
    # Fallback to direct LLM if retrieved context is weak/irrelevant
    if _should_fallback_to_llm(request.query):
        llm = Settings.llm
        try:
            comp = llm.complete(f"{_get_system_prompt()}\nПитання: {request.query}")
            return {"response": str(comp)}
        except Exception as e:
            # If direct call fails, fall back to RAG
            qe = _make_query_engine()
            resp = qe.query(request.query)
            return {"response": str(resp), "note": f"LLM fallback failed: {e}"}
    else:
        qe = _make_query_engine()
        resp = qe.query(request.query)
        return {"response": str(resp)}

@app.post("/refresh")
def refresh_index():
    global index
    # Backwards-compatible: full rebuild and activate
    idx_id, idx = _build_full_index()
    index = idx
    global active_index_id
    active_index_id = idx_id
    return {"status": "Переіндексовано та активовано", "index_id": idx_id}


# -----------------------------
# File upload and index control
# -----------------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save uploaded file into DATA_DIR
    if not file.filename:
        raise HTTPException(status_code=400, detail="Не задано ім'я файлу")
    dst_dir = Path(DATA_DIR)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / file.filename
    try:
        with open(dst_path, "wb") as f:
            content = await file.read()  # reading into memory; for large files, stream chunked
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка збереження файлу: {e}")
    return {"status": "Файл завантажено", "filename": file.filename, "path": str(dst_path)}


@app.post("/index/build")
def build_index_all_files():
    global index, active_index_id
    index_id, idx = _build_full_index()
    # Do not auto-apply here; just return id
    return {"status": "Індекс збудовано", "index_id": index_id}


@app.post("/index/apply")
def apply_index(req: ApplyIndexRequest):
    global index, active_index_id
    try:
        _configure_llm()
        idx = _load_index_by_id(req.index_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Індекс не знайдено")
    active_index_id = req.index_id
    index = idx
    return {"status": "Індекс активовано", "index_id": active_index_id}


# -----------------------------
# WebSocket streaming endpoint
# -----------------------------

def _get_system_prompt() -> str:
    return SYSTEM_PROMPT_VALUE


def _apply_text_qa_template():
    # Friendlier QA prompt in Ukrainian: use context if relevant; otherwise general knowledge
    Settings.text_qa_template = PromptTemplate(
        (
            f"{_get_system_prompt()}\n\n"
            "Тобі може бути надано контекст з документів.\n\n"
            "Контекст (може бути порожнім або нерелевантним):\n{context_str}\n\n"
            "Запит користувача: {query_str}\n\n"
            "Інструкції:\n"
            "- Якщо контекст має відношення до запиту — використай його.\n"
            "- Якщо контексту бракує або він нерелевантний — відповідай за загальними знаннями.\n"
            "- Відповідай українською, стисло та по суті."
        )
    )


def _build_prompt_from_history(history, message: str) -> str:
    """Compose a single prompt text from chat history and the new message.
    History format: [{"role": "user|assistant|system", "content": str}, ...]
    """
    parts = [f"System: {_get_system_prompt()}"]
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


def _configure_llm():
    """Configure Gemini LLM and Gemini embeddings via google-genai.
    LLM: LlamaIndex Gemini wrapper; Embeddings: custom class using google-genai.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY не задано у .env")
    model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    Settings.llm = GoogleGenAI(model=model, api_key=GOOGLE_API_KEY)
    Settings.embed_model = GeminiEmbedding(model=os.getenv("EMBED_MODEL", "gemini-embedding-001"), api_key=GOOGLE_API_KEY)

    _apply_text_qa_template()


def _make_query_engine():
    """Create a query engine with similarity cutoff to drop weak context."""
    return index.as_query_engine(
        similarity_top_k=RAG_TOP_K,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=RAG_CUTOFF)],
        streaming=False,
    )


def _should_fallback_to_llm(question: str) -> bool:
    """Decide whether to bypass RAG and answer directly with the LLM.

    Modes (via env `RAG_MODE`):
    - "rag_only": never fallback; always use storage index
    - "llm_only": always fallback; never use storage
    - "auto" (default): use retrieval confidence to decide; only fallback if weak
    """
    try:
        mode = (RAG_MODE or "auto").strip().lower()
        if mode == "rag_only":
            return False
        if mode == "llm_only":
            return True

        q = (question or "").strip().lower()
        # Only treat clear greetings as LLM-only in auto mode
        if re.search(r"\b(привіт|добрий день|вітаю|hello|hi|hey)\b", q):
            return True

        # Retrieval-based decision: if best similarity < cutoff, fallback
        retriever = index.as_retriever(similarity_top_k=RAG_TOP_K)
        nodes = retriever.retrieve(question)
        if not nodes:
            return True
        best = max((n.score or 0.0) for n in nodes)
        return best < RAG_CUTOFF
    except Exception:
        return True


class GeminiGenAIEmbedding(BaseEmbedding):
    """Embeddings via google-genai embed_content API."""
    def __init__(self, model: str = "gemini-embedding-001", api_key: Optional[str] = None):
        self.model = model
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _extract_embedding(self, result) -> List[float]:
        emb = getattr(result, "embeddings", None)
        if emb is None:
            emb = getattr(result, "embedding", None)
            if emb is None:
                return []
            vals = getattr(emb, "values", None)
            return list(vals) if vals is not None else list(emb)
        first = emb[0] if isinstance(emb, (list, tuple)) and emb else emb
        vals = getattr(first, "values", None)
        return list(vals) if vals is not None else list(first)

    def get_text_embedding(self, text: str) -> List[float]:
        res = self.client.models.embed_content(model=self.model, contents=text)
        return self._extract_embedding(res)

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_text_embedding(t) for t in texts]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Send a JSON handshake so client always receives valid JSON first
    await websocket.send_json({"type": "ready"})
    global index
    try:
        while True:
            # Expect a JSON text frame with {type: "ask", message: str, history: []}
            data_text = await websocket.receive_text()
            import json
            try:
                data = json.loads(data_text)
            except Exception:
                await websocket.send_json({
                    "type": "error",
                    "error": "Невірний формат повідомлення. Надсилайте JSON."
                })
                continue

            if data.get("type") != "ask":
                await websocket.send_json({
                    "type": "error",
                    "error": "Підтримується лише type=ask."
                })
                continue

            if index is None:
                await websocket.send_json({
                    "type": "error",
                    "error": "Індекс ще не готовий"
                })
                continue

            message = (data.get("message") or "").strip()
            history = data.get("history") or []
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "error": "Порожнє питання"
                })
                continue

            # Build a single prompt with history context
            prompt = _build_prompt_from_history(history, message)

            # Query with async streaming
            try:
                final_text = ""
                if _should_fallback_to_llm(message):
                    # Bypass RAG if context looks weak
                    comp = Settings.llm.complete(f"{_get_system_prompt()}\nПитання: {prompt}")
                    final_text = str(comp)
                    await websocket.send_json({"type": "token", "token": final_text})
                else:
                    aengine = index.as_query_engine(
                        streaming=True,
                        similarity_top_k=RAG_TOP_K,
                        node_postprocessors=[
                            SimilarityPostprocessor(
                                similarity_cutoff=RAG_CUTOFF
                            )
                        ],
                    )
                    # Use the user's latest message for retrieval
                    resp = await aengine.aquery(message)
                    final_text_tokens = []
                    async_gen_attr = getattr(resp, "async_response_gen", None)
                    if async_gen_attr is not None:
                        # Support both API shapes: property (async generator) or callable returning it
                        agen = async_gen_attr() if callable(async_gen_attr) else async_gen_attr
                        try:
                            async for token in agen:
                                final_text_tokens.append(token)
                                await websocket.send_json({"type": "token", "token": token})
                        except TypeError:
                            # In case it's sync generator exposed wrongly, fall back
                            pass
                    if not final_text_tokens:
                        # Fallback to sync generator if available
                        token_stream = getattr(resp, "response_gen", None)
                        if token_stream is not None:
                            for token in token_stream:
                                final_text_tokens.append(token)
                                await websocket.send_json({"type": "token", "token": token})
                    if final_text_tokens:
                        final_text = "".join(final_text_tokens)
                    else:
                        # No streaming support -> single chunk
                        final_text = str(resp)
                        await websocket.send_json({"type": "token", "token": final_text})
            except Exception as e:
                await websocket.send_json({"type": "error", "error": str(e)})
                continue

            # Return the updated history to the client
            new_history = list(history) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": final_text},
            ]
            await websocket.send_json({
                "type": "done",
                "message": final_text,
                "history": new_history,
            })
    except WebSocketDisconnect:
        # Client disconnected
        return


# -----------------------------
# Static HTML client
# -----------------------------

@app.get("/")
def serve_index():
    index_path = Path(__file__).parent / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html не знайдено")
    return FileResponse(index_path)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Enable CORS for using API from a different origin (for the HTML client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Utility endpoints: files and indexes listing
# -----------------------------

@app.get("/files")
def list_files():
    base = Path(DATA_DIR)
    base.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(base.iterdir() if base.exists() else [], key=lambda x: x.name.lower()):
        if p.is_file():
            stat = p.stat()
            items.append({
                "name": p.name,
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
            })
    return {"files": items}


@app.get("/index/list")
def list_indexes():
    items = []
    for idx_id, path in _list_index_dirs():
        items.append({
            "id": idx_id,
            "mtime": int(path.stat().st_mtime),
        })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"active": active_index_id, "items": items}


# -----------------------------
# Runtime config endpoints
# -----------------------------

@app.get("/config")
def get_config():
    return {
        "rag_mode": RAG_MODE,
        "rag_top_k": RAG_TOP_K,
        "rag_cutoff": RAG_CUTOFF,
        "system_prompt": SYSTEM_PROMPT_VALUE,
    }


@app.post("/config")
def set_config(update: ConfigUpdate):
    global RAG_MODE, RAG_TOP_K, RAG_CUTOFF, SYSTEM_PROMPT_VALUE
    if update.rag_mode is not None:
        mode = (update.rag_mode or "").strip().lower()
        if mode not in ("auto", "rag_only", "llm_only"):
            raise HTTPException(status_code=400, detail="Невірний rag_mode (auto|rag_only|llm_only)")
        RAG_MODE = mode
    if update.rag_top_k is not None:
        try:
            v = int(update.rag_top_k)
            if v < 1 or v > 50:
                raise ValueError
            RAG_TOP_K = v
        except Exception:
            raise HTTPException(status_code=400, detail="rag_top_k має бути цілим 1..50")
    if update.rag_cutoff is not None:
        try:
            v = float(update.rag_cutoff)
            if not (0.0 <= v <= 1.0):
                raise ValueError
            RAG_CUTOFF = v
        except Exception:
            raise HTTPException(status_code=400, detail="rag_cutoff має бути числом 0.0..1.0")
    if update.system_prompt is not None:
        SYSTEM_PROMPT_VALUE = str(update.system_prompt or DEFAULT_SYSTEM_PROMPT)
        # Re-apply QA template with new system prompt
        try:
            _apply_text_qa_template()
        except Exception:
            pass
    return get_config()

from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.api.routes import config as config_routes
from app.api.routes import files as files_routes
from app.api.routes import indexes as indexes_routes
from app.api.routes import query as query_routes
from app.api.routes import ws as ws_routes
from app.api.routes import suggest as suggest_routes
from app.core.config import AppSettings, RuntimeConfig
from app.core.indexing import latest_index_dir, load_index_by_id, build_full_index, has_supported_files
from app.core.llm import configure_llm
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    settings = AppSettings()
    # Init logging early
    setup_logging(settings)
    runtime = RuntimeConfig.from_settings(settings)

    app = FastAPI()

    # Attach state
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.index = None
    app.state.active_index_id = None

    # Middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    @app.middleware("http")
    async def add_request_id(request, call_next):
        rid = str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        try:
            response.headers["X-Request-ID"] = rid
        except Exception:
            pass
        return response

    # Static files and root
    static_dir = settings.base_dir / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def serve_index():
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="index.html не знайдено")
        return FileResponse(index_path)

    # Routers
    api_router.include_router(query_routes.router)
    api_router.include_router(files_routes.router)
    api_router.include_router(indexes_routes.router)
    api_router.include_router(config_routes.router)
    api_router.include_router(ws_routes.router)
    api_router.include_router(suggest_routes.router)
    app.include_router(api_router)

    @app.on_event("startup")
    def on_startup():
        # Configure LLM
        configure_llm(settings, runtime)
        # Load latest index or build one
        latest = latest_index_dir(settings.persist_base_path)
        if latest is None:
            # If there’s no persisted index yet, only build if there are files
            if has_supported_files(settings.data_path):
                idx_id, idx = build_full_index(settings, runtime)
                app.state.active_index_id = idx_id
                app.state.index = idx
            else:
                # Defer indexing until files are present
                app.state.active_index_id = None
                app.state.index = None
        else:
            idx_id, _ = latest
            app.state.active_index_id = idx_id
            app.state.index = load_index_by_id(settings.persist_base_path, idx_id)

    return app


app = create_app()

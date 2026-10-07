"""FastAPI application factory.

Layering (top -> bottom):

    api/routes   — HTTP contracts (thin: parse request, call service, return)
    services     — business logic / orchestration        (Phase 2+)
    nlp/         — NLP components                        (Phase 2+)
    memory/, rag/— conversation memory, retrieval         (Phase 6-7)
    models/      — SQLAlchemy entities                    (Phase 6+)
    schemas/     — Pydantic request/response contracts    (exists now)
    core/        — logging, exceptions                   (exists now)

Run with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.nlp.nltk_data import ensure_nltk_data

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook — later phases load models & DB pools here."""
    settings = get_settings()
    logger.info("starting %s v%s (%s)", settings.app_name, __version__, settings.env)

    # Phase 2: make sure NLTK corpora exist (network only if actually missing).
    try:
        fetched = ensure_nltk_data()
        if fetched:
            logger.info("downloaded NLTK resources: %s", ", ".join(fetched))
    except Exception:
        logger.warning(
            "could not ensure NLTK data — pipeline will use safe fallbacks. "
            "Fix manually: python -m app.nlp.nltk_data",
            exc_info=True,
        )

    yield
    logger.info("shutdown complete")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="NOVA — intelligent NLP chatbot. Phase 1: project foundation.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": settings.app_name, "version": __version__, "docs": "/docs"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)

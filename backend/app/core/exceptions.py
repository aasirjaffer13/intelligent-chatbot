"""Application-specific exceptions and their FastAPI handlers.

Pattern: raise ``NovaError`` subclasses deep inside services/NLP code, and the
handler turns them into clean JSON responses — no stack traces leak to clients,
while full details still land in the logs.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class NovaError(Exception):
    """Base class for all expected NOVA errors."""

    status_code: int = 500
    code: str = "nova_error"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFoundError(NovaError):
    status_code = 404
    code = "not_found"


class ExternalServiceError(NovaError):
    """Upstream dependency (LLM provider, database, ...) failed."""

    status_code = 502
    code = "external_service_error"


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON error handlers to the application."""

    @app.exception_handler(NovaError)
    async def _nova_error_handler(request: Request, exc: NovaError) -> JSONResponse:
        logger.warning("NovaError on %s %s: %s", request.method, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "detail": exc.detail}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "detail": "Internal server error"}},
        )

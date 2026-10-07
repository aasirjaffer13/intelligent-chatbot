"""GET /api/health — liveness probe used by the frontend status indicator."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse, summary="Service liveness check")
async def health() -> HealthResponse:
    logger.debug("health check requested")
    return HealthResponse(status="healthy", service="nova")

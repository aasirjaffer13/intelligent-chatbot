"""Single aggregation point for all API routes.

Adding a new endpoint = create a route module under ``routes/`` + one import
and ``include_router`` line here. The FastAPI app never imports routes directly.
"""

from fastapi import APIRouter

from app.api.routes import chat, conversations, documents, health, status

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(conversations.router)
api_router.include_router(status.router)

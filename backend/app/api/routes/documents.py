"""Document management endpoints (Phase 7 RAG).

Thin HTTP layer: parse the upload, call ``DocumentService``, map typed
errors to status codes (415/400/413), return contract schemas.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.rag import (
    DocumentService,
    EmptyDocumentError,
    UnsupportedFileType,
    get_document_service,
)
from app.schemas import DocumentInfoResponse

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post(
    "/upload",
    response_model=DocumentInfoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document for RAG",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, .txt, or .md"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentInfoResponse:
    from app.config import get_settings

    data = await file.read()
    limit = get_settings().max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file exceeds {get_settings().max_upload_mb} MB limit",
        )
    try:
        info = service.ingest(file.filename or "upload.bin", data, file.content_type or "")
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc))
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"could not process upload: {exc}",
        )
    return DocumentInfoResponse(
        id=info.id,
        filename=info.filename,
        content_type=info.content_type,
        size_bytes=info.size_bytes,
        chunk_count=info.chunk_count,
        created_at=info.created_at,
    )


@router.get("", response_model=list[DocumentInfoResponse], summary="List uploaded documents")
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentInfoResponse]:
    return [
        DocumentInfoResponse(
            id=info.id,
            filename=info.filename,
            content_type=info.content_type,
            size_bytes=info.size_bytes,
            chunk_count=info.chunk_count,
            created_at=info.created_at,
        )
        for info in service.list()
    ]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> None:
    if not service.delete(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")

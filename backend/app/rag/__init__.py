"""RAG package (Phase 7): chunking, storage, retrieval, grounded answers.

Public surface used by the API routes and the chat pipeline::

    from app.rag import (
        DocumentService, Retriever, get_rag_store, get_document_service,
        compose_grounded_answer,
    )
"""

from app.rag.answerer import GroundedAnswer, compose_grounded_answer
from app.rag.chunker import Chunk, chunk_text
from app.rag.extractor import EmptyDocumentError, UnsupportedFileType, extract_text
from app.rag.service import (
    DocumentService,
    Retriever,
    get_document_service,
    get_rag_store,
    get_retriever,
)
from app.rag.store import DocumentInfo, RagStore, ScoredChunk

__all__ = [
    "Chunk",
    "DocumentInfo",
    "DocumentService",
    "EmptyDocumentError",
    "GroundedAnswer",
    "RagStore",
    "Retriever",
    "ScoredChunk",
    "UnsupportedFileType",
    "chunk_text",
    "compose_grounded_answer",
    "extract_text",
    "get_document_service",
    "get_rag_store",
    "get_retriever",
]

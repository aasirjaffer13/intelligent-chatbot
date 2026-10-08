"""Text extraction from uploaded documents (Phase 7).

Supported today: PDF (pypdf), plain text, Markdown. Everything else is
rejected with a typed error the route maps to HTTP 415 — refusing unknown
formats is a *feature* (garbage in, garbage embeddings out).
"""

from __future__ import annotations

import io


class UnsupportedFileType(ValueError):
    """Upload type not supported — message is safe to show to the user."""


class EmptyDocumentError(ValueError):
    """No extractable text (e.g. scanned/image-only PDF)."""


_SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
_SUPPORTED_TYPES = {"text/plain", "text/markdown", "application/pdf"}


def extract_text(filename: str, data: bytes) -> str:
    """Return the document's text content or raise a typed error."""
    lowered = (filename or "").lower()

    if lowered.endswith(".pdf"):
        text = _extract_pdf(data)
    elif any(lowered.endswith(ext) for ext in _SUPPORTED_TEXT_EXTENSIONS):
        text = data.decode("utf-8", errors="replace")
    elif _looks_like_text_mime(filename, data):
        # Content-Type said text/plain even without a known extension.
        text = data.decode("utf-8", errors="replace")
    else:
        raise UnsupportedFileType(
            f"unsupported file type for {filename!r} — use PDF, .txt, or .md"
        )

    text = text.strip()
    if not text:
        raise EmptyDocumentError(
            f"no extractable text in {filename!r} (scanned images need OCR)"
        )
    return text


def _looks_like_text_mime(filename: str, data: bytes) -> bool:
    # Heuristic fallback: small ASCII-ish payloads are treated as text.
    if not data:
        return False
    sample = data[:1024]
    return b"\x00" not in sample and all(b < 128 for b in sample)


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf raises many malformed-PDF types
        raise UnsupportedFileType(f"could not read PDF: {exc}") from exc
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)

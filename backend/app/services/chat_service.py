"""Chat orchestration — the brain of the API layer.

Flow (Phase 2):

    ChatRequest -> preprocessing -> rule-based reply -> ChatResponse

Each arrow is a future extension point:
    Phase 3 inserts intent classification after preprocessing
    Phase 4 inserts entity extraction
    Phase 5 inserts embedding-based retrieval
    Phase 8 swaps the reply generator for an LLM behind LLMProvider

The route stays a one-liner forever; all evolution happens here.
"""

from __future__ import annotations

import logging
from time import perf_counter

from app.nlp.preprocessing import PreprocessConfig, PreprocessResult, preprocess
from app.schemas import ChatRequest, ChatResponse, ProcessingInfo

logger = logging.getLogger(__name__)

MAX_TOKEN_PREVIEW = 10


class ChatService:
    """Stateless chat pipeline (state arrives with memory in Phase 6)."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def handle(self, request: ChatRequest) -> ChatResponse:
        """Run the pipeline and build the API response."""
        started = perf_counter()
        result = preprocess(request.message, self.config)
        elapsed_ms = (perf_counter() - started) * 1000

        logger.info(
            "preprocessed %d chars -> %d tokens in %.1f ms",
            len(request.message),
            len(result.tokens),
            elapsed_ms,
        )
        logger.debug("normalized=%r tokens=%s", result.normalized_text, result.tokens)

        return ChatResponse(
            response=self._generate_reply(result),
            intent="unknown",  # Phase 3 replaces this with real classification
            confidence=0.0,
            entities=[],
            processing=ProcessingInfo(
                tokens=result.tokens,
                normalized_text=result.normalized_text,
                sentences=result.sentences,
            ),
            session_id=request.session_id,
        )

    def _generate_reply(self, result: PreprocessResult) -> str:
        """Phase 2 reply generator: an honest diagnostic (no intent yet).

        This is deliberately rule-based — response *content* gets intelligent
        in Phase 3 (templates per intent) and Phase 8 (LLM)."""
        if not result.tokens:
            return (
                "I normalized your message but found no content tokens "
                "(only stopwords/punctuation/URLs were left). "
                "Intent classification arrives in Phase 3."
            )

        preview = ", ".join(result.tokens[:MAX_TOKEN_PREVIEW])
        if len(result.tokens) > MAX_TOKEN_PREVIEW:
            preview += ", …"
        count = len(result.tokens)
        return (
            f"Preprocessed your message into {count} token{'s' if count != 1 else ''}: "
            f"{preview}. Intent classification arrives in Phase 3."
        )


# Module-level singleton: the service holds config, not request state.
chat_service = ChatService()

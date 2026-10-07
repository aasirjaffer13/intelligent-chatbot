"""Chat orchestration — the brain of the API layer.

Pipeline (Phase 3):

    ChatRequest
      -> preprocessing            (Phase 2)
      -> intent classification     (Phase 3)  -> label + confidence
      -> rule-based reply          (deterministic templates / real time)
      -> ChatResponse

Extension points already reserved:
    Phase 4 inserts entity extraction after intent
    Phase 7 answers document intents via RAG
    Phase 8 swaps reply generation for an LLM (intent still drives routing)
    Phase 9 adds tool decisions (calculator, time, weather, ...)
"""

from __future__ import annotations

import logging
from datetime import datetime
from time import perf_counter

from app.nlp.intent import IntentResult, classify_intent, load_dataset
from app.nlp.preprocessing import PreprocessConfig, PreprocessResult, preprocess
from app.schemas import ChatRequest, ChatResponse, ProcessingInfo

logger = logging.getLogger(__name__)


class ChatService:
    """Stateless chat pipeline (persistent memory arrives in Phase 6)."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def handle(self, request: ChatRequest) -> ChatResponse:
        """Run the pipeline and build the API response."""
        started = perf_counter()
        processed = preprocess(request.message, self.config)
        intent_result = classify_intent(request.message)
        elapsed_ms = (perf_counter() - started) * 1000

        logger.info(
            "intent=%s conf=%.2f (%s) | %d tokens | %.1f ms",
            intent_result.label,
            intent_result.confidence,
            intent_result.method,
            len(processed.tokens),
            elapsed_ms,
        )

        return ChatResponse(
            response=self._generate_reply(processed, intent_result),
            intent=intent_result.label,
            confidence=round(intent_result.confidence, 4),
            entities=[],  # Phase 4 fills this
            processing=ProcessingInfo(
                tokens=processed.tokens,
                normalized_text=processed.normalized_text,
                sentences=processed.sentences,
            ),
            session_id=request.session_id,
        )

    # --- response generation (deterministic; LLM takes over in Phase 8) ------

    def _generate_reply(self, processed: PreprocessResult, intent: IntentResult) -> str:
        if intent.label == "time":
            return _time_reply()

        responses = load_dataset().responses_for(intent.label)
        if not responses:
            return (
                "I'm not sure how to respond to that yet."
                if intent.label == "unknown"
                else "OK."
            )

        # Deterministic template choice: stable across retries, varied by input.
        reply = responses[len(processed.original_text) % len(responses)]
        if intent.label == "unknown":
            return f"{reply} (intent confidence: {intent.confidence:.2f})"
        return reply


def _time_reply() -> str:
    """Deterministic, always-correct answer — no LLM needed (routing principle)."""
    now = datetime.now().astimezone()
    return (
        f"It's {now.strftime('%H:%M')} on {now.strftime('%A, %B %d, %Y')} "
        f"(timezone {now.strftime('%Z')})."
    )


# Module-level singleton: the service holds config, not request state.
chat_service = ChatService()

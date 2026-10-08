"""Chat orchestration — the brain of the API layer.

Pipeline (Phase 6):

    ChatRequest
      -> memory: resolve session, load context window   (Phase 6)
      -> preprocessing                                  (Phase 2)
      -> intent classification                           (Phase 3)
      -> entity extraction                               (Phase 4)
      -> reply: memory-aware + deterministic templates
      -> persist both turns to memory                    (Phase 6)
      -> ChatResponse

Extension points already reserved:
    Phase 7 answers document intents via RAG
    Phase 8 swaps reply generation for an LLM (intent still drives routing)
    Phase 9 adds tool decisions (calculator, time, weather, ...)
"""

from __future__ import annotations

import logging
from datetime import datetime
from time import perf_counter

from app.memory import (
    MemoryContext,
    MemoryStore,
    build_context,
    extract_name,
    get_memory_store,
    is_name_question,
)
from app.nlp.entities import EntitySpan, get_entity_extractor
from app.nlp.intent import IntentResult, classify_intent, load_dataset
from app.nlp.preprocessing import PreprocessConfig, PreprocessResult, preprocess
from app.schemas import ChatRequest, ChatResponse, Entity, ProcessingInfo

logger = logging.getLogger(__name__)


class ChatService:
    """Chat pipeline with pluggable conversation memory."""

    def __init__(
        self,
        config: PreprocessConfig | None = None,
        memory: MemoryStore | None = None,
        window: int | None = None,
    ) -> None:
        if window is None:
            from app.config import get_settings

            window = get_settings().memory_window
        self.config = config or PreprocessConfig()
        self.memory = memory if memory is not None else get_memory_store()
        self.window = window
        self.extractor = get_entity_extractor()

    def handle(self, request: ChatRequest) -> ChatResponse:
        """Run the pipeline and build the API response."""
        started = perf_counter()

        # Memory first: context must reflect history BEFORE this turn lands.
        session_id = self.memory.get_or_create_conversation(request.session_id)
        recent = self.memory.recent_messages(session_id, self.window)
        context = build_context(recent, request.message)

        processed = preprocess(request.message, self.config)
        intent_result = classify_intent(request.message)
        entity_spans = self.extractor.extract(request.message)
        reply = self._generate_reply(processed, intent_result, context, request.message)
        elapsed_ms = (perf_counter() - started) * 1000

        # Persist both turns — next request's context reads them back.
        self.memory.add_message(
            session_id, "user", request.message, intent_result.label, intent_result.confidence
        )
        self.memory.add_message(
            session_id, "assistant", reply, intent_result.label, intent_result.confidence
        )

        logger.info(
            "intent=%s conf=%.2f (%s) | %d tokens | %d entities | "
            "turn=%d history=%d | %.1f ms",
            intent_result.label,
            intent_result.confidence,
            intent_result.method,
            len(processed.tokens),
            len(entity_spans),
            context.turn_count,
            len(recent),
            elapsed_ms,
        )

        return ChatResponse(
            response=reply,
            intent=intent_result.label,
            confidence=round(intent_result.confidence, 4),
            entities=[_to_schema_entity(span) for span in entity_spans],
            processing=ProcessingInfo(
                tokens=processed.tokens,
                normalized_text=processed.normalized_text,
                sentences=processed.sentences,
            ),
            session_id=session_id,
        )

    # --- response generation (deterministic; LLM takes over in Phase 8) ------

    def _generate_reply(
        self,
        processed: PreprocessResult,
        intent: IntentResult,
        context: MemoryContext,
        raw_message: str,
    ) -> str:
        # Memory-aware answers come first: they use REAL stored history.
        if is_name_question(raw_message):
            if context.known_name:
                return (
                    f"Your name is {context.known_name} — you told me earlier "
                    "in our conversation."
                )
            return 'I don\'t know your name yet — say "My name is ..." and I\'ll remember.'
        declared = extract_name(raw_message)
        if declared:
            return f"Nice to meet you, {declared}! I'll remember that for this session."

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


def _to_schema_entity(span: EntitySpan) -> Entity:
    """Map the NLP-layer span onto the stable API contract."""
    return Entity(
        text=span.text,
        label=span.label,
        start=span.start,
        end=span.end,
        confidence=round(span.confidence, 4),
    )


# Module-level singleton: the service holds config, not request state.
chat_service = ChatService()

"""Chat orchestration — the brain of the API layer.

Pipeline (Phase 8):

    ChatRequest
      -> memory: resolve session, load context window   (Phase 6)
      -> preprocessing                                  (Phase 2)
      -> intent classification                           (Phase 3)
      -> entity extraction                               (Phase 4)
      -> reply: RAG for document questions (Phase 7),
                agent loop: LLM + tool calls (Phases 8-9),
                deterministic cases + template fallback
      -> persist both turns to memory                    (Phase 6)
      -> ChatResponse

Extension points already reserved:
    Phase 10 adds streaming + frontend polish
"""

from __future__ import annotations

import logging
from datetime import datetime
from time import perf_counter
from typing import Any

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
from app.rag import RagStore, compose_grounded_answer
from app.schemas import ChatRequest, ChatResponse, Entity, ProcessingInfo, Source

logger = logging.getLogger(__name__)

# Sentinel: "no llm argument given" (resolve lazily) vs llm=None ("disabled").
_LAZY: Any = object()


class ChatService:
    """Chat pipeline with pluggable memory, RAG store and LLM provider."""

    def __init__(
        self,
        config: PreprocessConfig | None = None,
        memory: MemoryStore | None = None,
        window: int | None = None,
        rag_store: RagStore | None = None,
        llm: Any = _LAZY,
        agent: Any = _LAZY,
    ) -> None:
        if window is None:
            from app.config import get_settings

            window = get_settings().memory_window
        self.config = config or PreprocessConfig()
        self.memory = memory if memory is not None else get_memory_store()
        self.window = window
        self._rag_store = rag_store  # None = resolve lazily on first document question
        self._llm = None if llm is _LAZY else llm
        self._llm_resolved = llm is not _LAZY  # injected provider/None == resolved
        self._agent = None if agent is _LAZY else agent
        self._agent_resolved = agent is not _LAZY
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

        # RAG answers document questions from retrieved context only (Phase 7).
        sources: list[Source] = []
        rag_reply: str | None = None
        if intent_result.label == "document_question":
            rag = self._rag_answer(request.message)
            if rag is not None:
                rag_reply, sources = rag

        reply = rag_reply or self._generate_reply(
            processed,
            intent_result,
            context,
            request.message,
            entity_spans=entity_spans,
        )
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
            sources=sources,
            processing=ProcessingInfo(
                tokens=processed.tokens,
                normalized_text=processed.normalized_text,
                sentences=processed.sentences,
            ),
            session_id=session_id,
        )

    # --- RAG (Phase 7) -------------------------------------------------------

    def _rag_answer(self, message: str) -> tuple[str, list[Source]] | None:
        """Ground a document question in retrieved chunks.

        Returns None when RAG itself fails — the caller falls back to the
        template reply rather than dropping the message on the floor.
        """
        from app.config import get_settings
        from app.rag import Retriever, get_rag_store
        from app.services.embedding_service import get_embedding_service

        try:
            if self._rag_store is None:
                self._rag_store = get_rag_store()
            settings = get_settings()
            documents = self._rag_store.list_documents()

            if not documents:
                answer = compose_grounded_answer(
                    message, [], min_score=settings.rag_min_score, has_documents=False
                )
            else:
                hits = Retriever(self._rag_store, get_embedding_service()).search(
                    message, top_k=settings.rag_top_k
                )
                answer = compose_grounded_answer(
                    message,
                    hits,
                    min_score=settings.rag_min_score,
                    has_documents=True,
                )
            return answer.reply, answer.sources
        except Exception:
            logger.exception("RAG failed — falling back to template reply")
            return None

    # --- LLM + agent reply generation (Phases 8-9) ----------------------------

    def _get_llm(self) -> Any:
        """Resolve the configured provider once; None = templates only."""
        if not self._llm_resolved:
            from app.llm import get_llm_provider

            self._llm = get_llm_provider()
            self._llm_resolved = True
        return self._llm

    def _get_agent(self) -> Any:
        """Resolve the agent loop once: injected agent wins; otherwise the
        loop wraps the resolved provider (None provider = no agent)."""
        if self._agent_resolved:
            return self._agent
        provider = self._get_llm()
        if provider is None:
            self._agent_resolved = True
            return None
        from app.agent import AgentLoop
        from app.config import get_settings
        from app.tools import build_default_registry

        self._agent = AgentLoop(
            provider, build_default_registry(), max_steps=get_settings().agent_max_steps
        )
        self._agent_resolved = True
        return self._agent

    def _llm_reply(
        self,
        *,
        intent: IntentResult,
        context: MemoryContext,
        raw_message: str,
        entity_spans: list[EntitySpan],
    ) -> str | None:
        """Run the agent loop (Phase 9): the model may call tools and
        iterate, or answer directly (single step, Phase 8 behaviour).

        Returns None when there is no agent or anything fails — the
        caller falls back to deterministic templates (availability over
        fluency; a provider outage must not break chat).
        """
        from app.config import get_settings
        from app.llm import LLMError, build_chat_request

        try:
            agent = self._get_agent()
            if agent is None:
                return None
            settings = get_settings()
            request = build_chat_request(
                message=raw_message,
                intent_label=intent.label,
                confidence=intent.confidence,
                entity_texts=[f"{s.label}={s.text}" for s in entity_spans],
                messages=context.messages,
                known_name=context.known_name,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
            result = agent.run(request)
            text = (result.reply or "").strip()
            if not text:
                logger.warning("agent returned an empty reply — using template")
                return None
            if result.used_tools:
                logger.info(
                    "agent reply via tools: %s | %d steps",
                    ", ".join(dict.fromkeys(result.used_tools)),
                    len(result.steps),
                )
            else:
                logger.debug("agent direct reply (%d steps)", len(result.steps))
            return text
        except LLMError as exc:
            logger.warning("LLM failed (%s) — falling back to template reply", exc)
            return None
        except Exception:
            logger.exception("unexpected agent failure — falling back to template reply")
            return None

    # --- reply generation: deterministic cases, LLM phrasing, templates ------

    def _generate_reply(
        self,
        processed: PreprocessResult,
        intent: IntentResult,
        context: MemoryContext,
        raw_message: str,
        entity_spans: list[EntitySpan] | None = None,
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

        # Always-correct computed answers never go to a model (routing principle).
        if intent.label == "time":
            return _time_reply()

        # Phase 8: the LLM words the reply; the pipeline already decided
        # what it means. Any failure degrades to templates below.
        llm_reply = self._llm_reply(
            intent=intent,
            context=context,
            raw_message=raw_message,
            entity_spans=entity_spans or [],
        )
        if llm_reply:
            return llm_reply

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

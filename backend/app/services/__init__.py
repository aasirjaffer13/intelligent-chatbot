"""Business-logic services (orchestration between API and NLP/DB layers).

``chat_service`` is the single place that runs the pipeline:
message -> preprocessing -> (Phase 3: intent) -> (Phase 4: entities) -> reply.
"""

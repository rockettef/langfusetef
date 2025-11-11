# Telco Agent V6 — Roadmap & Refactor Plan

Status: proposal / planning

Goal
----
Refactor the Telco agent into a modular, stateful system that supports back-and-forth, multi-turn conversations while keeping full Langfuse observability, per-turn scoring, cost & credit controls, streaming responses, and human handoff. The V6 design should be easy to test (unit + integration), deploy, and scale.

Success criteria
----------------
- Multi-turn conversations with session state persisted in-memory (with Redis optional backing).
- Each user message and agent response becomes an observable Langfuse event/trace with token/cost metadata.
- Per-turn evaluations (LLM-as-judge) attach scores to traces and are recorded in the UI.
- Credit/limit enforcement prevents runaway costs and can be tuned per deployment.
- Streaming responses supported for low-latency UX; partial traces or incremental updates allowed.
- Human handoff endpoint and workflow for low-confidence or escalated cases.

High-level approach
-------------------
1. Break the single-file script into small modules with clear responsibilities (Langfuse wrapper, cost/credit, conversation manager, evaluator, prompt manager, storage, API server).
2. Implement a ConversationManager that holds session history, trace IDs, and coordinates Langfuse observations and LLM calls.
3. Move side-effects (Langfuse SDK calls, DB writes, S3 interactions) behind thin wrappers so they can be mocked in tests.
4. Add a small FastAPI HTTP interface for interactive testing and integration with frontends/operators.
5. Provide a CLI demo (`telco-agent-test-v6.py`) that runs an interactive shell for manual testing.

Suggested file layout
---------------------
langfuse-telco-test/
- agent/
  - __init__.py
  - core.py               # ConversationManager, Agent orchestration
  - langfuse_client.py    # Thin wrapper around Langfuse SDK (observations, scoring, trace updates)
  - cost.py               # cost calculation, credit checks, session accounting
  - eval.py               # evaluation logic (LLM-as-judge) and scheduler
  - prompt_manager.py     # centralize prompts; fetch from Langfuse if enabled
  - storage.py            # session/message store (in-memory + Redis optional)
- server/
  - api.py                # FastAPI endpoints: /message, /session, /handoff
- workers/
  - evaluator_worker.py   # optional background worker for scoring
- tests/
  - unit/
  - integration/
- telco-agent-test-v6.py  # CLI runner and minimal PoC
- V6-ROADMAP.md
- CHANGELOG.md

Key components & contracts
---------------------------
- ConversationManager
  - start_session(user_id, session_id=None) -> Session
  - receive_user_message(session_id, text, metadata={}) -> Message
  - generate_response(session_id, message_id) -> Response (sync or generator for streaming)
  - attach_score(session_id, message_id, score)

- Langfuse wrapper (langfuse_client)
  - start_observation(as_type, name, metadata) -> observation handle
  - update_observation(handle, usage_details|partial_text)
  - finish_observation(handle)
  - score_current_trace(name, value, comment)
  - update_current_trace(metadata)
  - start_session_trace(session_id, user_id)

- Cost & Credit (cost)
  - calculate_cost(input_tokens, output_tokens)
  - estimate_cost_for_operation(op_name)
  - check_and_reserve_budget(session_id, estimated_cost) -> bool
  - commit_cost(session_id, actual_cost, tokens)

Langfuse integration pattern
----------------------------
For each user message:
1. start an observation of type `message` (user-message) and include metadata (session_id, user_id, message_id).
2. start a nested observation for `generation` when invoking the LLM.
3. after generation, update the generation observation with usage_details (prompt_tokens, completion_tokens, total_tokens) and estimated cost.
4. finish nested observations, then optionally schedule an async evaluation and attach score via `score_current_trace` when it completes.
5. update session-level trace metadata with cumulative cost/tokens.

Streaming & incremental traces
-----------------------------
- If your LLM client supports token streaming, forward tokens to the end-user and buffer them on the server.
- Periodically update the active `generation` observation with partial_text (or write final text when generation completes) so Langfuse shows in-progress activity.

Evaluation & scoring
--------------------
- Use a small evaluator module that:
  - sends the judge prompt to the LLM (or pushes it to a worker) and robustly extracts numeric scores (regex + bounds checking);
  - normalizes to 0–1 and calls `score_current_trace` inside the observation context when possible;
  - falls back to logging+queueing if the trace context is not available.

Human handoff
-------------
- Provide `server/api.py` endpoint to flag a session for handoff.
- ConversationManager should expose a hook like `on_handoff(session_id, reason)` which notifies operator UI (webhook or writes a message to Redis/DB).

Testing & CI
------------
- Unit tests (fast): cost calculations, evaluator parsing, prompt manager, ConversationManager logic (in-memory store)
- Integration tests (CI): run a small Docker Compose stack (Postgres, MinIO, Redis) with mocked LLM responses (or a lightweight mocking server). Verify Langfuse client wrapper is called (use a fake client).
- Add GitHub Actions job: run unit tests, lint, and an optional integration matrix using lightweight mocks.

Migration plan (step-by-step)
---------------------------
1. Create `agent/cost.py` and move cost functions from v5 there. Add unit tests.
2. Create `agent/langfuse_client.py` and move all Langfuse-related calls into it. Add a fake Langfuse client for tests.
3. Implement `agent/prompt_manager.py` with current prompts as defaults.
4. Implement `agent/storage.py` (in-memory session store) and write tests for history management.
5. Implement `agent/core.py` with ConversationManager that uses the above modules; wire minimal end-to-end flow in `telco-agent-test-v6.py` (CLI).
6. Add `server/api.py` with a `/message` endpoint that calls ConversationManager; test locally.
7. Add background evaluator worker or simple in-process async evaluator.
8. Iterate on streaming and human handoff features.




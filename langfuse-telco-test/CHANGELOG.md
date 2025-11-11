# CHANGELOG — langfuse-telco-test

This file summarizes the incremental features and fixes added to the Telco test harness across versions. Use this as a quick reference when picking a test script or preparing release notes.

## v5 — telco-agent-test-v5.py (Nov 11, 2025)
- Single-model strategy: gpt-5-nano as the canonical deployment for cost efficiency
- Token budgets per operation (classification, generation, evaluation)
- Credit limit checks (daily/monthly) with pre-checks to skip expensive operations
- Cost calculation and session-level cost tracking; attached to traces
- Reasoning-token monitoring and warnings; safeguards for token exhaustion
- Robust evaluation flow that attaches normalized quality scores to current traces
- Session metadata enriched with version/model/credit flags

## v4 — telco-agent-test-v4.py (Nov 5, 2025)
- Fixed evaluation scoring integration: uses `score_current_trace()` to reliably attach scores inside `@observe()` contexts
- Improved error handling and logging across the workflow
- Performance metrics and improved context management for evaluations
- Better score creation using SDK v3 methods and more predictable trace attachments

## v3 — telco-agent-test-v3.py (Oct 30, 2025)
- Full Langfuse observability introduced: hierarchical traces, `start_as_current_generation`, and explicit token usage mapping
- LLM-as-judge evaluation (score via LLM) and API call to attach scores to traces (`langfuse.score` / explicit trace_id)
- Session tracking via `update_current_trace()` and dataset creation helpers (`create_dataset`, `create_dataset_item`)
- Prompt management hooks and dataset-driven testing examples

## v2 — telco-agent-test-v2.py
- Enhancements over baseline: added debug outputs, safer empty-response handling, and larger `max_completion_tokens` limits for some calls
- Still focused on core classification + generation with Langfuse instrumentation

## v1 — telco-agent-test.py (baseline)
- Basic Azure OpenAI integration (classification + generation)
- Basic Langfuse instrumentation using `@observe`
- Minimal example and smoke-test coverage

## Notes
- The canonical, most up-to-date test harness is `telco-agent-test-v5.py`. Use the specific file matching the feature set you need (e.g., cost tracking vs simple smoke tests).
- For CI and reproducibility, consider adding `langfuse-telco-test/requirements.txt` that pins `langfuse`, `openai`, and `python-dotenv` versions used during development.



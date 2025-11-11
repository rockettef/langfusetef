# langfuse-telco-test
# langfuse-telco-test

This folder contains integration and end-to-end test artifacts for the Langfuse Telco demo and the supporting helper scripts used during development.

This README documents the test harness, how it uses the Langfuse SDK and the additional capabilities we've integrated while developing the Telco agent.

## What this test suite covers
- End-to-end LLM workflows using Azure OpenAI (classification + generation).
- Full Langfuse observability: hierarchical traces, generation blocks, token usage and cost tracking.
- LLM-as-judge scoring (automated quality evaluation) with scores attached to traces.
- Session tracking and per-user/session metadata captured in traces.
- S3-backed batch exports and event uploads (MinIO in local dev setup).
- Media upload capability and path-style S3 compatibility for MinIO.
- Dataset management: create datasets and dataset items for automated evaluation and regression tests.
- Prompt management integration: ability to wire prompts from Langfuse UI into the agent.
- Ingestion to ClickHouse + PostgreSQL (Prisma) for persistence and analytics.

## Files in this folder
- `telco-agent-test-v5.py` — example end-to-end test harness demonstrating classification, generation, evaluation (LLM judge), session tracking and Langfuse SDK usage.
- `test.py` — minimal smoke test that exercises Langfuse SDK tracing and flush behavior.
- `README.md` — this file.
- Add `tests/`, `fixtures/` subfolders for any new integration tests or sample inputs.

## Quick architecture and runtime notes
- The test clients run locally (Python) and send traces/events to the Langfuse web + worker services running in Docker Compose.
- Local object storage is provided by MinIO (S3-compatible). The app writes batch exports and events to S3 (MinIO) before the worker ingests them.
- Services used in `docker-compose.yml`: `postgres`, `clickhouse`, `redis`, `minio`, `langfuse-web`, `langfuse-worker`.
- For S3/MinIO compatibility, path-style addressing is enabled and credentials are configured in `.env`.

## Environment (important)
- Copy or review the repository root `.env` file for the environment variables used by tests and services. Key variables used by the tests include:
	- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`
	- `LANGFUSE_S3_*_ACCESS_KEY_ID`, `LANGFUSE_S3_*_SECRET_ACCESS_KEY`, `LANGFUSE_S3_*_ENDPOINT`
	- `DATABASE_URL`, `DIRECT_URL` (Postgres)
	- `CLICKHOUSE_URL`, `CLICKHOUSE_MIGRATION_URL`
	- `REDIS_HOST`, `REDIS_PORT`, `REDIS_AUTH`

Note: `.env` may contain secrets and is usually not committed to source control. Keep it secure and do not push secrets to a public repo.

## Local quickstart
1. Start required services from the repository root:

```powershell
cd "c:\Users\id05367\OneDrive - Telefonica\Documentos\git\langfuse"
docker compose up -d
```

2. Activate the Python venv (PowerShell example):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\venv\Scripts\Activate.ps1
```

3. Install/update Python deps if needed (venv):

```powershell
pip install -r langfuse-telco-test/requirements.txt
# or: pip install python-dotenv openai langfuse
```

4. Run the quick smoke test:

```powershell
python .\langfuse-telco-test\test.py
```

5. Run the full telco demo/test (longer):

```powershell
python .\langfuse-telco-test\telco-agent-test-v3.py
```

6. Tail logs while running tests to observe ingestion/score events and S3 uploads:

```powershell
docker logs --tail 200 -f langfuse-minio-1
docker logs --tail 200 -f langfuse-langfuse-web-1
docker logs --tail 200 -f langfuse-langfuse-worker-1
```

## How scoring (LLM judge) works in the tests
- The demo uses an LLM-based judge prompt to rate generated responses on relevance, accuracy, tone and completeness.
- Scores are normalized to 0–1 and the test harness calls `langfuse.score(trace_id, name, value, comment)` to attach scores to the active trace.
- If the trace id is not available in the current context, the client resolves the current trace id (if supported) before attaching the score. This avoids noisy failures when running outside an instrumented trace.

## Datasets and automated evaluation
- Use `langfuse.create_dataset()` and `langfuse.create_dataset_item()` from the Python SDK (examples embedded in `telco-agent-test-v3.py`) to create evaluation datasets.
- Recommended workflow:
	1. Create dataset programmatically or in the Langfuse UI
	2. Add representative test cases (input, expected output, metadata)
	3. Run the agent over the dataset and collect scores
	4. Use the Langfuse UI to compare runs, filter by intent, and analyze regressions

## Prompt management
- You can move hard-coded system prompts to Langfuse prompt management for centralized editing and versioning.
- Workflow:
	1. Create prompts in the Langfuse UI (Settings → Prompts)
	2. Tag prompts (e.g., `production`, `staging`)
	3. Retrieve prompts at runtime with `langfuse.get_prompt()` and compile them with test inputs

## Troubleshooting
- If you see S3 upload errors (500 / "Failed to upload JSON to S3") inside containers, ensure the S3 endpoints point to the MinIO service name on the Docker network (e.g., `http://minio:9000`) — not `localhost:9090`.
- After changing `.env`, recreate containers so they pick up the new values:

```powershell
docker compose down
docker compose up -d --force-recreate --renew-anon-volumes
```

- If scoring is not appearing in the UI:
	- Confirm `LANGFUSE_S3_BATCH_EXPORT_ENABLED=true` and `LANGFUSE_S3_*` creds are correct.
	- Check worker logs for `batch-export` and `evaluation-execution` queue messages.
	- Verify traces exist in the UI (`/traces`) and that `trace_id` is being propagated from the SDK calls.

## Development notes & best practices
- Keep tests self-contained and independent. Avoid relying on manual state.
- When adding tests that hit external LLMs, mock or record responses where practical to avoid quota/cost surprises.
- For CI, provide a `docker-compose.test.yml` that uses a lightweight or mocked LLM endpoint and reuses the same `postgres/clickhouse/minio/redis` services.

## Contact
If you need help integrating new checks or adding dataset-driven regression tests, open an issue or reach out to the Langfuse engineering team in your workspace.

---

Happy testing — the telco demo is instrumented to give you detailed observability, scoring and dataset-driven evaluation so you can iterate on prompts, models and prompts quickly.

Guidance:
- Add tests, fixtures and helper scripts under this folder.
- Keep tests self-contained and avoid relying on manual state.
- If tests require local services, use the repository's Docker Compose stack (see top-level `docker-compose.yml`).

Suggested files to add:
- `tests/` - integration test cases
- `fixtures/` - test fixtures and sample data
- `docker-compose.test.yml` - optional compose override for test-specific services

How to run (example):
1. Start services:

```powershell
cd "c:\\Users\\id05367\\OneDrive - Telefonica\\Documentos\\git\\langfuse"
docker compose up -d
```

2. Run tests (example using npm/yarn/pnpm):

```powershell
pnpm -w test
```

Replace the commands above with your project's test runner and scripts.

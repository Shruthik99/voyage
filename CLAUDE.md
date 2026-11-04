# Voyage

Agentic US travel planner. An LLM picks tools and repairs plans; plain-Python code
validates every plan. See docs/PLAN.md, docs/DECISIONS.md, and docs/SPEC.md.

## Hard rules
- US only. Input is a list of "City, ST" strings; never split one string on commas.
  Reject non-US or malformed input with LocationError.
- Never hide a failure or fabricate a successful result. Tools raise typed errors
  (ToolError, LocationError) and never return default, placeholder, or made-up data.
  Callers may retry, use an explicit backup source, ask the user, or return a result
  flagged incomplete (warnings, verified=False). A partial result is fine if it is labeled.
- Every place carries source and source_id from a tool result in the current run.
- The validator is plain Python, never an LLM. The LLM proposes; the validator decides.
- Secrets live in .env only. Do not open, read, print, or edit .env. App code loads it
  with python-dotenv. Use .env.example for variable names.
- No paid APIs. LLM = OpenAI SDK against Groq's endpoint, default model openai/gpt-oss-120b.

## Layout
src/ (parsing, tools, models, geo, constants, llm, pricing, trace, planner, validator, agent), tests/, docs/,
traces/ (JSON run traces, gitignored), cache/ (API response cache, gitignored)

## Code style
- Python 3.13. Type hints on public functions.
- Pydantic v2 models for data crossing a boundary (API responses, LLM output, itinerary).
- httpx with explicit timeouts. Retry only transient errors (timeout, 429, 5xx).
- Small functions. Ask before adding any new dependency.

## Testing
- Every module gets tests. Every bug fix adds a regression test.
- Tests never call live APIs; use cached fixtures or mocks. The evals/ runner is a script,
  not a test, and calls live APIs by design; its own tests (tests/test_evals_*.py) are
  dry-run only.
- Run `uv run pytest` and `uv run ruff check .` before saying a task is done.

## Workflow
- Do not run git commit or git push. The user reviews and commits.
- For validator.py and agent.py, propose a plan first and wait for approval.
- Windows PowerShell: no && chaining, no touch. One command per line.
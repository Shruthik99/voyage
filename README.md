# Voyage: an agentic US trip planner with a validator that decides

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Pydantic](https://img.shields.io/badge/Pydantic-v2-green) ![LLM](https://img.shields.io/badge/LLM-gpt--oss--120b%20on%20Groq-orange) ![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

A tool-calling LLM proposes multi-day itineraries for US cities. Eight plain-Python rules decide whether a plan is acceptable, and failures go back to the model for up to three repairs. The project is evaluated in three arms on 30 trips.

This is a portfolio project, not a product. See "Limits".

## Problem
AI trip planners produce plans that look complete but are never checked: stops in the wrong state, place ids that no tool returned, requested interests skipped. This project asks whether a rule-based validator plus a repair loop measurably improves plan quality over letting the model write the plan unchecked.

## Approach
1. Code geocodes every route city (a city that fails stops the run early), fetches a 16-day forecast, and builds the day skeleton from route x days per stop.
2. **Tool phase:** the model calls `geocode` and `search_places` (Geoapify) to collect real places.
3. **Proposal phase:** the model names place ids from a catalog; code builds each stop from the tool result, so coordinates, address and state are never retyped by a model.
4. **Validator (no LLM):** V1 location, V2 budget, V3 interests, V4 no placeholders, V5 every (source, id) came from a tool result this run, V6 daily distance, V7 no duplicates, V8 stops per day.
5. **Repair:** only failing results go back to the model, at most 3 times; otherwise the plan is returned flagged `verified=False`.

The two model phases are separate because Groq rejects tools and a strict JSON schema in one request (HTTP 400, checked by hand against the live API).

## Evaluation
All arms use the same validator, the same gathered places, `gpt-oss-120b` on Groq, and low reasoning effort.

| Arm | What it is |
|---|---|
| A | LLM only, unrepaired: the model writes whole stops; nothing repairs a failure |
| B | ids, no repair: the model names ids (arm C's first attempt) |
| C | full system: ids plus up to 3 repairs |

- **30 trips:** single-, two- and three-city routes, ambiguous names (Portland OR/ME, Springfield, Columbus, Kansas City), a small town, and one invalid input that must be rejected before any API call.
- **Metrics:** share of trips passing all rules, hallucinated-venue rate (V5 failures per proposed stop), wrong-location stops (V1), interest coverage, repairs, tool calls, tokens and seconds.
- **Paired headline:** only trips where all three arms finished. B and C are paired, not independent.
- Counts are shown next to percentages. With about 30 trips, a gap of a few trips may not be real.

## Engineering
- Typed models (Pydantic v2) at every boundary: tool results, proposals, itineraries, validation reports.
- Tools raise typed errors and never return defaults or placeholders; a failure is reported, not hidden.
- Retries only for transient failures, with explicit limits; a rate limit waits as long as the provider asks.
- Resumable eval runner: one line per finished trial, completed trials skipped on restart, stops cleanly at a token budget.
- JSON trace per run (tool calls, validator reports, repairs, token usage); keys are scrubbed before anything is written.
- 288 automated tests (no live API calls), Ruff clean, and every design choice recorded in `docs/DECISIONS.md`.

## Results
Fill from `evals/results.md` (paired headline table).

| Metric | A | B | C |
|---|---|---|---|
| Trips passing all rules | [x of N] | [x of N] | [x of N] |
| Hallucinated-venue rate | [x of N stops] | [x of N stops] | [x of N stops] |
| Wrong-location stops | [x of N] | [x of N] | [x of N] |
| Interest coverage | [x of N] | [x of N] | [x of N] |
| Repairs per trip | 0 | 0 | [ ] |
| Tokens per trip (this arm only) | [ ] | [ ] | [ ] |

**What the runs showed**
- Arm A's failed ids were real place ids with altered characters, not invented places (a hand check of the failed ids found a near match for each among the cached real ids).
- At border cities (Kansas City MO/KS) the radius search returns places in the neighboring state; V1 catches them.
- **Where repair did not help:** Columbus, GA fails V3 in every arm because the place data returns no history sites there, and a Kansas City KS trip still failed after three repairs. Repairs also cost roughly twice the tokens of arm B.
- Groq sometimes rejects a generation outright (two error wordings); the retry covers both, and affected trips were rerun.
- One three-city trip failed once when the model dropped the state from a tool call; a rerun passed. Reported as a nondeterministic slip, not a fix.

**Bugs that live runs found (mocked tests had missed them)**
- Forecast coverage was judged by the local clock, not the destination's date.
- A forecast day with a null temperature caused the whole response to be rejected.
- Arm A scored 100% hallucinated only because it was never shown the `source` label.

## Limits
- **Prices are unknown.** No free source provides them, so V2 can warn but never fail.
- No weather replan, scheduler or live monitoring. This is not an autonomous system.
- "Hiking" means parks and outdoor spots, not trails.
- US only; city names are matched strictly ("St. Paul" is rejected, "Saint Paul" works).
- Free-tier limits (8,000 tokens per minute, 200,000 per day), so the evaluation ran over several days.
- One model, one provider, a small sample.

## Reproduction
```
uv sync
# .env (see .env.example): GEOAPIFY_KEY=...  GROQ_API_KEY=...
uv run pytest
uv run ruff check .
```
Regenerate the results table from the committed rows (no API keys needed):
```
uv run python -m evals.summarize --results evals/results.jsonl --out evals/results.md
```
Re-run the campaign (resumes where it stopped):
```
uv run python -m evals.run --reasoning-effort low
```
Dry run with a fake client, no network: `uv run python -m evals.run --dry-run --limit 3 --out results.tmp.jsonl --trace-dir traces/tmp`

Plan one trip:
```python
from datetime import date
from src.agent import TripRequest, plan_trip

req = TripRequest(route=["Denver, CO"], days_per_stop=2, travelers=2,
                  budget_usd=400.0, meal_allowance_usd=40.0,
                  interests=["museums", "hiking"], start_date=date(2026, 11, 10))
result = plan_trip(req, reasoning_effort="low")
print(result.itinerary.verified, result.report.warnings)
```

# Voyage

An agentic US trip planner where the model proposes and plain Python decides. An LLM gathers real places with tools and proposes an itinerary; eight rule checks (no LLM) validate it; failures go back to the model for up to three repairs. Most of the project is the evaluation: three arms, 30 trips, one fair comparison.

This is a portfolio project, not a product. See "Limits" for what it does not do.

## What it does
- **Input:** a route of "City, ST" stops, days per stop, travelers, budget, interests, start date.
- **Output:** a day-by-day itinerary of real places (Geoapify), a 16-day forecast (Open-Meteo), a validator report, and a JSON trace of every tool call and repair.
- US cities only. Malformed or non-US input is rejected before any API call.

## How it works
1. Code geocodes every route city (a city that fails stops the run early), fetches forecasts, and builds the day skeleton from route x days per stop.
2. **Tool phase:** the model calls `geocode` and `search_places` to collect real places.
3. **Proposal phase:** the model names place ids from a catalog; code builds each stop from the tool result, so coordinates, address and state are never retyped by a model.
4. **Validator:** eight pure-Python rules (below).
5. **Repair:** failing results go back to the model, at most 3 times; otherwise the plan is returned flagged `verified=False`.

The two model phases are separate because Groq rejects tools and a strict JSON schema in one request (HTTP 400, checked by hand against the live API).

## Validator rules
| Rule | Checks |
|---|---|
| V1 | each stop is in its day's city state and within a radius of the city |
| V2 | budget: `unpriced` (warn) / `within` / `uncertain` (warn) / `over` (fail) |
| V3 | every requested interest has a stop |
| V4 | no placeholders: name, source and id present |
| V5 | every (source, id) came from a tool result in this run |
| V6 | daily straight-line distance under a cap |
| V7 | no place used twice |
| V8 | each day has a sensible number of stops |

## Evaluation
All arms are scored by the same validator, on the same gathered places, with `gpt-oss-120b` on Groq at low reasoning effort.

| Arm | What it is |
|---|---|
| A | LLM only, unrepaired: the model writes whole stops; nothing repairs a failure |
| B | ids, no repair: the model names ids; this is arm C's first attempt |
| C | full system: ids, plus up to 3 repairs |

B and C are paired, not independent. 30 trips: 29 plannable (single-, two- and three-city routes, ambiguous names such as Portland OR/ME, a small town) and one invalid input that must be rejected before any API call.

### Results
Fill from `evals/results.md` (headline table, paired trips). Show counts next to percentages.

| Metric | A | B | C |
|---|---|---|---|
| Trips passing all checks | [x of N] | [x of N] | [x of N] |
| Hallucinated-venue rate (stops) | [x of N] | [x of N] | [x of N] |
| Wrong-location stops | [x of N] | [x of N] | [x of N] |
| Tokens per trip (own) | [ ] | [ ] | [ ] |

N is small (about 30 trips), so differences of a few trips may not be real. Re-create the table with `uv run python -m evals.summarize`.

## What the runs showed
- Arm A's failed ids are real place ids with altered characters, not invented places (checked against the cached tool responses).
- At border cities (Kansas City MO/KS) the radius search returns places in the neighboring state, which V1 catches; repairs reduced this but did not remove it.
- One trip fails V3 in every arm because the place data returns no history sites near Columbus, GA.
- Groq sometimes rejects a generation outright; the retry covers two wordings of that error.
- One three-city trip failed once when the model dropped the state from a tool call; a rerun passed. Reported as a nondeterministic slip, not a fix.

## Bugs that live runs found (mocked tests missed all of these)
- Forecast coverage judged by the local clock, not the destination's date.
- A forecast day with a null temperature caused the whole response to be rejected.
- Arm A scored 100% hallucinated only because it was never shown the `source` label.
- API-key text could reach an error message.

## Limits
- **Prices are unknown.** No free source gives them, so V2 can warn but never fail; pricing coverage is zero.
- **No weather replan** and no scheduler or live disruption monitoring. Not autonomous.
- "Hiking" means parks and outdoor spots, not trails.
- US only; city names are matched strictly ("St. Paul" is rejected, "Saint Paul" works).
- Free-tier limits: 8,000 tokens/min and 200,000 tokens/day, so the evaluation ran over several days.
- One model, one provider, small sample.

## Run it
    uv sync
    # .env: GEOAPIFY_KEY=... and GROQ_API_KEY=...
    uv run pytest
    uv run python -m evals.run --reasoning-effort low
    uv run python -m evals.summarize --results evals/results.jsonl --out evals/results.md

The runner appends one line per finished trial and skips completed ones, so it resumes across days.

## Layout
`src/` (parsing, tools, models, validator, agent, llm, trace, pricing) · `evals/` (trips, runner, metrics, summary, results) · `tests/` · `docs/` (SPEC, PLAN, DECISIONS)

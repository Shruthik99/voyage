# Eval results

Generated from `results.jsonl`: 90 row(s) across 30 trip(s).

1 invalid trip(s) rejected before any API call; excluded from every rate below.

No arm runs without the validator -- it scores all three. Arm A is **"LLM only, unrepaired"**: the model writes whole stops and the validator scores them, but nothing repairs a failure. Arm B is arm C's own first attempt with repair switched off, so B and C are paired, not independent samples.

## Headline: paired trips (all three arms finished)

Per-arm N: A=29, B=29, C=29. Paired N: 29.

| Metric | A (LLM only, unrepaired) | B (ids, no repair) | C (full system) |
|---|---|---|---|
| Pass rate | 41.4% | 48.3% | 65.5% |
| Hallucinated-venue rate (V5 untraceable, pooled) | 5.1% | 1.6% | 0.4% |
| Wrong-location stops (pooled) | 10.6% | 5.3% | 9.6% |
| Interest coverage (pooled) | 98.1% | 98.1% | 98.1% |
| Repairs per trip | 0.00 | 0.00 | 1.24 |
| Tool calls per trip | 8.03 | 8.07 | 8.07 |
| Tokens per trip (total, incl. shared gather) | 6,709 | 5,438 | 9,705 |
| Tokens per trip (this arm only) | 3,734 | 2,442 | 6,709 |
| Seconds per trip (model + tool time) | 60.19 | 42.48 | 86.77 |
| Average daily km | 4.22 | 2.10 | 0.99 |
| Tool-step cap hit rate | 0.0% | 0.0% | 0.0% |
| Estimated cost share | 0.0% | 0.0% | 0.0% |
| Unknown-cost stops per trip | 8.76 | 8.38 | 8.62 |
| Priced share | 0.0% | 0.0% | 0.0% |
| N | 29 | 29 | 29 |

### Failures by rule (paired trips)

| Rule | A | B | C |
|---|---|---|---|
| V1 | 20.7% (27 detail(s)) | 17.2% (13 detail(s)) | 20.7% (24 detail(s)) |
| V2 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V3 | 3.4% (1 detail(s)) | 3.4% (1 detail(s)) | 3.4% (1 detail(s)) |
| V4 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V5 | 37.9% (13 detail(s)) | 13.8% (4 detail(s)) | 3.4% (1 detail(s)) |
| V6 | 3.4% (1 detail(s)) | 3.4% (1 detail(s)) | 0.0% (0 detail(s)) |
| V7 | 10.3% (7 detail(s)) | 10.3% (4 detail(s)) | 17.2% (15 detail(s)) |
| V8 | 13.8% (20 detail(s)) | 17.2% (14 detail(s)) | 6.9% (10 detail(s)) |

## All finished trips (unpaired; not comparable across arms)

| Metric | A (LLM only, unrepaired) | B (ids, no repair) | C (full system) |
|---|---|---|---|
| Pass rate | 41.4% | 48.3% | 65.5% |
| Hallucinated-venue rate (V5 untraceable, pooled) | 5.1% | 1.6% | 0.4% |
| Wrong-location stops (pooled) | 10.6% | 5.3% | 9.6% |
| Interest coverage (pooled) | 98.1% | 98.1% | 98.1% |
| Repairs per trip | 0.00 | 0.00 | 1.24 |
| Tool calls per trip | 8.03 | 8.07 | 8.07 |
| Tokens per trip (total, incl. shared gather) | 6,709 | 5,438 | 9,705 |
| Tokens per trip (this arm only) | 3,734 | 2,442 | 6,709 |
| Seconds per trip (model + tool time) | 60.19 | 42.48 | 86.77 |
| Average daily km | 4.22 | 2.10 | 0.99 |
| Tool-step cap hit rate | 0.0% | 0.0% | 0.0% |
| Estimated cost share | 0.0% | 0.0% | 0.0% |
| Unknown-cost stops per trip | 8.76 | 8.38 | 8.62 |
| Priced share | 0.0% | 0.0% | 0.0% |
| N | 29 | 29 | 29 |

### Failures by rule (all finished trips)

| Rule | A | B | C |
|---|---|---|---|
| V1 | 20.7% (27 detail(s)) | 17.2% (13 detail(s)) | 20.7% (24 detail(s)) |
| V2 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V3 | 3.4% (1 detail(s)) | 3.4% (1 detail(s)) | 3.4% (1 detail(s)) |
| V4 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V5 | 37.9% (13 detail(s)) | 13.8% (4 detail(s)) | 3.4% (1 detail(s)) |
| V6 | 3.4% (1 detail(s)) | 3.4% (1 detail(s)) | 0.0% (0 detail(s)) |
| V7 | 10.3% (7 detail(s)) | 10.3% (4 detail(s)) | 17.2% (15 detail(s)) |
| V8 | 13.8% (20 detail(s)) | 17.2% (14 detail(s)) | 6.9% (10 detail(s)) |

## Errors

Infrastructure failures (AgentError / LLMUnavailable), reported separately so one is never counted as a model failure in disguise:

A: 0, B: 0, C: 0

## Known gaps

- Pricing coverage and estimated share are structurally 0.0: `src/pricing.py` is a documented stub that returns `UNPRICED` for every stop.
- The budget trips (V2) are untested: with no stop priced, V2 can only warn, never fail.
- No weather-replan metric is reported: no replan function exists in `src/agent.py` yet.

# Eval results

Generated from `results.jsonl`: 42 row(s) across 14 trip(s).

0 invalid trip(s) rejected before any API call; excluded from every rate below.

No arm runs without the validator -- it scores all three. Arm A is **"LLM only, unrepaired"**: the model writes whole stops and the validator scores them, but nothing repairs a failure. Arm B is arm C's own first attempt with repair switched off, so B and C are paired, not independent samples.

## Headline: paired trips (all three arms finished)

Per-arm N: A=14, B=14, C=14. Paired N: 14.

| Metric | A (LLM only, unrepaired) | B (ids, no repair) | C (full system) |
|---|---|---|---|
| Pass rate | 35.7% | 57.1% | 78.6% |
| Hallucinated-venue rate (V5 untraceable, pooled) | 9.6% | 2.1% | 0.0% |
| Wrong-location stops (pooled) | 4.3% | 4.3% | 2.2% |
| Interest coverage (pooled) | 96.0% | 96.0% | 96.0% |
| Repairs per trip | 0.00 | 0.00 | 0.93 |
| Tool calls per trip | 5.57 | 5.57 | 5.57 |
| Tokens per trip (total, incl. shared gather) | 5,047 | 3,993 | 5,943 |
| Tokens per trip (this arm only) | 2,968 | 1,914 | 3,864 |
| Seconds per trip (model + tool time) | 45.40 | 35.76 | 56.20 |
| Average daily km | 1.33 | 1.35 | 1.04 |
| Tool-step cap hit rate | 0.0% | 0.0% | 0.0% |
| Estimated cost share | 0.0% | 0.0% | 0.0% |
| Unknown-cost stops per trip | 6.71 | 6.57 | 6.43 |
| Priced share | 0.0% | 0.0% | 0.0% |
| N | 14 | 14 | 14 |

### Failures by rule (paired trips)

| Rule | A | B | C |
|---|---|---|---|
| V1 | 14.3% (4 detail(s)) | 14.3% (4 detail(s)) | 7.1% (2 detail(s)) |
| V2 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V3 | 7.1% (1 detail(s)) | 7.1% (1 detail(s)) | 7.1% (1 detail(s)) |
| V4 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V5 | 57.1% (9 detail(s)) | 0.0% (2 detail(s)) | 0.0% (0 detail(s)) |
| V6 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V7 | 7.1% (3 detail(s)) | 7.1% (1 detail(s)) | 7.1% (1 detail(s)) |
| V8 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |

## All finished trips (unpaired; not comparable across arms)

| Metric | A (LLM only, unrepaired) | B (ids, no repair) | C (full system) |
|---|---|---|---|
| Pass rate | 35.7% | 57.1% | 78.6% |
| Hallucinated-venue rate (V5 untraceable, pooled) | 9.6% | 2.1% | 0.0% |
| Wrong-location stops (pooled) | 4.3% | 4.3% | 2.2% |
| Interest coverage (pooled) | 96.0% | 96.0% | 96.0% |
| Repairs per trip | 0.00 | 0.00 | 0.93 |
| Tool calls per trip | 5.57 | 5.57 | 5.57 |
| Tokens per trip (total, incl. shared gather) | 5,047 | 3,993 | 5,943 |
| Tokens per trip (this arm only) | 2,968 | 1,914 | 3,864 |
| Seconds per trip (model + tool time) | 45.40 | 35.76 | 56.20 |
| Average daily km | 1.33 | 1.35 | 1.04 |
| Tool-step cap hit rate | 0.0% | 0.0% | 0.0% |
| Estimated cost share | 0.0% | 0.0% | 0.0% |
| Unknown-cost stops per trip | 6.71 | 6.57 | 6.43 |
| Priced share | 0.0% | 0.0% | 0.0% |
| N | 14 | 14 | 14 |

### Failures by rule (all finished trips)

| Rule | A | B | C |
|---|---|---|---|
| V1 | 14.3% (4 detail(s)) | 14.3% (4 detail(s)) | 7.1% (2 detail(s)) |
| V2 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V3 | 7.1% (1 detail(s)) | 7.1% (1 detail(s)) | 7.1% (1 detail(s)) |
| V4 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V5 | 57.1% (9 detail(s)) | 0.0% (2 detail(s)) | 0.0% (0 detail(s)) |
| V6 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |
| V7 | 7.1% (3 detail(s)) | 7.1% (1 detail(s)) | 7.1% (1 detail(s)) |
| V8 | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) | 0.0% (0 detail(s)) |

## Errors

Infrastructure failures (AgentError / LLMUnavailable), reported separately so one is never counted as a model failure in disguise:

A: 0, B: 0, C: 0

## Known gaps

- Pricing coverage and estimated share are structurally 0.0: `src/pricing.py` is a documented stub that returns `UNPRICED` for every stop.
- The budget trips (V2) are untested: with no stop priced, V2 can only warn, never fail.
- No weather-replan metric is reported: no replan function exists in `src/agent.py` yet.

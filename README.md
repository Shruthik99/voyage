# voyage

Agentic US travel planner. An LLM picks tools and repairs plans; plain-Python code
validates every plan. See docs/SPEC.md, docs/DECISIONS.md, and docs/PLAN.md.

## Evaluation

`evals/` runs a three-arm comparison over `evals/trips.json` (30 US trips), scored by the
same plain-Python validator (`src/validator.py`) for every arm:

- **A ("LLM only, unrepaired")** -- the model writes whole stops; the validator scores
  them but nothing repairs a failure.
- **B ("ids, no repair")** -- arm C's own first attempt, read from its trace, not run.
- **C ("full system")** -- the model names stops by id; up to 3 repair attempts.

```
uv run python -m evals.run --limit 2 --reasoning-effort low   # a small live smoke run
uv run python -m evals.summarize                              # evals/results.jsonl -> evals/results.md
```

Numbers are not published here yet: they come from `evals/results.md`, generated after
the full live campaign described in `docs/DECISIONS.md`, which has not been run.

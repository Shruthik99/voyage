# Decisions

Each entry: what we chose, what we rejected, why.

## US only, country config later
Chose: US destinations, with country-specific logic behind a config object.
Rejected: global support now.
Why: the US has free official data (NWS, NPS, MBTA). India needs different adapters later, added without a rewrite.

## Validator is plain Python, not an LLM
Chose: rule-based checks, each pass/fail.
Rejected: LLM-as-judge.
Why: checks must be repeatable and cannot be talked into passing a bad plan.

## Fail loudly at the tool layer, degrade visibly at the agent layer
Chose: tools raise errors; the agent retries, asks, or returns flagged partial results.
Rejected: silent fallback values.
Why: a wrong answer that looks right is worse than an honest partial one.

## Hand-written agent loop, no framework
Chose: plain Python loop (plan, tool calls, validate, repair, max 3 tries).
Rejected: LangGraph or similar, for now.
Why: small enough to explain line by line. Revisit if persistence or approvals need it.

## OpenAI SDK against Groq, openai/gpt-oss-120b
Chose: OpenAI-compatible client pointed at Groq's free tier.
Rejected: paid APIs, provider-specific SDKs.
Why: $0 budget, tool calling support, and switching providers later is a config change.

## Classic ML for geometry, LLM for judgment (planned)
Chose: k-means to group places into days, nearest-neighbor to order stops; the LLM only chooses tools and repairs plans.
Rejected: asking the LLM to group and order places.
Why: geometry is math, and LLM-only grouping is how wrong-city stops happen.

## No RAG
Chose: live API lookups only.
Rejected: vector store over place or park text.
Why: no document-search problem exists yet. Revisit for a park advisor tool.

## Files now, SQLite when the monitor exists
Chose: JSON traces and a disk cache for API responses.
Rejected: Postgres, a vector database.
Why: no persistent state is needed until the scheduler, and the cache protects free-tier quotas and makes evals repeatable.

## Real triggers where free data exists, simulated otherwise
Chose: live weather (NWS or Open-Meteo); closures injected as labeled test events.
Rejected: implying live closure feeds.
Why: no free API reports surprise closures, so honesty about the simulation matters more.
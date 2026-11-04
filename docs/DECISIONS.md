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

## Two-phase LLM calls, forced by Groq
Chose: a tool-calling phase (tools, no schema), then a proposal phase (strict json_schema, no tools).
Rejected: one request that does both.
Why: Groq returns HTTP 400 ("json mode cannot be combined with tool/function calling"), and gpt-oss-120b does not call tools in parallel. Checked by hand against the live API.

## The LLM names place ids; code builds the stops
Chose: the model returns source_ids from a catalog, and code fills in coordinates, address and state from the tool result. stop_mode="full" lets the model write whole stops, for the eval baseline only.
Rejected: the model writing stops.
Why: a retyped coordinate or source cannot be trusted, and ids make V5 structurally true. The baseline arm still shows what an unvalidated planner does.

## Weather is fetched by code and judged by the dates returned
Chose: code fetches 16 days of forecast per route city before the model runs, skips days with missing values, and explains each uncovered day from the dates Open-Meteo actually returned.
Rejected: letting the model call get_forecast, and computing coverage from the machine's date.
Why: two live bugs. Open-Meteo counts days from the destination's local date (Denver was a day behind), and the 16th day came back with a null temperature. Both passed the mocked tests.

## Three arms, one shared tool phase
Chose: a single gather() per trip (geocode, forecast, the tool-calling loop) backs two proposals,
arm A (stop_mode="full", max_repairs=0) and arm C (stop_mode="ids", max_repairs=3). Each arm's
trace adopts the gather's events, so it stays a self-contained record of the tool results every
stop traces back to.
Rejected: each arm running its own tool phase; persisting catalogs to disk across days.
Why: a separate tool phase per arm would let the comparison differ in two variables at once
(stop_mode and which places the model happened to find), and would make arm A a weaker, not
fairer, baseline, since it would be handed a worse catalog than arm C. Persisting the registry
across days would mean a later proposal reads tool results from a *previous* run, which breaks
"every place carries source and source_id from a tool result in the current run" outright.

## Arm B is read from arm C's trace, not run
Chose: arm B ("ids, no repair") is arm C's own event stream, truncated at and including the
first validation event -- not a separate LLM run.
Rejected: running a third, independent stop_mode="ids" / max_repairs=0 conversation.
Why: arm C's first proposal already is that conversation; running it again would spend tokens
to answer a question the trace already answers, and would turn B-vs-C into two independent
samples instead of the same first attempt with and without repair. The trade is that B and C
are paired, not independent: fine for comparing them to each other, not for a significance
claim that assumes independence.

## Arm A sees the full record, not just an id
Chose: in stop_mode="full", the catalog entry for each place includes source, source_id, name,
address, state, lat and lon -- every field ProposalStopFull requires -- and the system prompt
tells the model to copy those fields exactly, never alter an id, and never write a place not in
the catalog. stop_mode="ids" is untouched: no source or address field, no extra tokens.
Rejected: leaving the full-mode catalog at the ids-mode summary (source_id, name, interest,
state, lat, lon) and trusting the model to supply source correctly on its own.
Why: the first live smoke test showed arm A scoring a 0% real-venue rate -- not because it
invented places, but because it was never told the source label and wrote "catalog", the name
of the JSON section it was copying from. V5 correctly failed every stop. The baseline must be
given what it needs to copy correctly, or it measures a setup flaw instead of the model; V5
itself is unchanged, so a genuinely altered id or an invented source still fails.
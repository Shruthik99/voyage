# Voyage spec

## Input
- route: list of "City, ST" stops, each parsed by parse_route
- days_per_stop, travelers, budget_usd (total), interests (subset of: museums, history,
  hiking, food), start_date

## Output: Itinerary
Days, each with ordered stops. Every stop has: name, address, lat, lon, state, interest,
source, source_id, cost_low_usd, cost_high_usd, cost_source, cost_confidence. Each day has
a forecast, or a warning saying weather is unverified. The itinerary has a warnings list.

## Cost (layered; each stop records where its number came from)
1. A real price from a source (OSM charge tag, NPS entrance fee): low = high, confidence high
2. Known free (no-fee flag): 0 to 0, confidence high
3. Otherwise a low-high range by place type, confidence low, labeled "estimate"
4. A traveler-entered price overrides everything above
The itinerary reports estimated_share: the fraction of total cost that comes from layer 3.
Ranges are documented with how they were chosen. Revise after pricing research.

## Validator rules (plain Python, each pass/fail, failures name the rule, day and stop)
V1 Location: each stop's state matches its day's city state (a None state falls back to
   the distance check and adds a warning), and the stop is within RADIUS_KM of the
   city's geocoded point.
V2 Budget: with S_high = sum(cost_high) x travelers and S_low = sum(cost_low) x travelers:
   S_high <= budget_usd passes; S_low <= budget_usd < S_high passes with an "uncertain"
   warning; S_low > budget_usd fails.
V3 Interests: every requested interest has at least one stop in the trip.
V4 No placeholders: name is non-empty, source and source_id are non-empty.
V5 Traceability: every (source, source_id) appears in a tool result from this run.
V6 Daily distance: total straight-line km per day <= MAX_DAY_KM.
V7 No duplicates: a source_id appears at most once in the trip.
V8 Day size: each day has between MIN_STOPS and MAX_STOPS stops.

## Agent behavior
Tools: geocode, search_places, get_forecast. The LLM chooses tools and proposes the
itinerary; the validator decides. On failure, failures go back to the LLM. Maximum 3
repair attempts, then return the result flagged unverified. Every tool call and
validator result is written to a JSON trace.

## Weather replan
If a day's rain_chance >= RAIN_THRESHOLD and it has outdoor (hiking) stops, replace only
those stops with indoor ones (museums, history, food), keep every other day unchanged,
then re-validate.

## Done means
Metrics for N US trips, with and without the validator: pass rate, repairs per trip,
tool calls per trip, average daily km, and the share of stops kept after a weather replan.
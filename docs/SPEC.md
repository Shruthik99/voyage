# Voyage spec

## Input
- route: list of "City, ST" stops, each parsed by parse_route
- days_per_stop, travelers, budget_usd (total), meal_allowance_usd (per person per day),
  interests (subset of: museums, history, hiking, food), start_date

## Output: Itinerary
Days, each with ordered stops. Every stop has: name, address, lat, lon, state, interest,
source, source_id, cost_low_usd, cost_high_usd (null if unknown), cost_basis
(per_person | per_group | per_vehicle), cost_source, cost_confidence
(known | estimate | unknown). Each day has a forecast, or a warning saying weather is
unverified. The itinerary has a warnings list, estimated_share, and unknown_count.

## Cost (layered; each stop records where its number came from)
1. Known price from a source (NPS entranceFees, OSM charge tag): low = high, confidence known
2. Known free (no-fee flag): 0 to 0, confidence known
3. Food: the traveler's meal allowance, confidence estimate
4. Other types: a low-high range from a documented sample of real prices (method and
   sample in docs/PRICING.md), confidence estimate. No sample means unknown.
5. A traveler-entered price overrides everything above
Group total: per_person costs x travelers; per_group and per_vehicle costs counted once.
No price is ever invented. Unknown stays unknown.

## Validator rules (plain Python, each pass/fail, failures name the rule, day and stop)
V1 Location: each stop's state matches its day's city state (a None state falls back to
   the distance check and adds a warning), and the stop is within RADIUS_KM of the
   city's geocoded point.
V2 Budget, with S_low and S_high the group totals of cost_low and cost_high:
   incomplete: any stop's cost is unknown (warning, lists the stops)
   over budget: S_low > budget_usd (fail)
   uncertain: S_low <= budget_usd < S_high (pass with warning)
   within budget: S_high <= budget_usd (pass; labeled "within estimated budget" unless
   estimated_share is 0)
V3 Interests: every requested interest has at least one stop in the trip.
V4 No placeholders: name, source and source_id are non-empty.
V5 Traceability: every (source, source_id) appears in a tool result from this run.
V6 Daily distance: total straight-line km per day <= MAX_DAY_KM.
V7 No duplicates: a source_id appears at most once in the trip.
V8 Day size: each day has between MIN_STOPS and MAX_STOPS stops.

## Agent behavior
Tools: geocode, search_places, get_forecast. The LLM chooses tools and proposes the
itinerary; the validator and a plain-Python pricing step produce the numbers. The LLM
never supplies prices. On failure, failures go back to the LLM. Maximum 3 repair
attempts, then return the result flagged unverified. Every tool call and validator
result is written to a JSON trace.

## Weather replan
If a day's rain_chance >= RAIN_THRESHOLD and it has outdoor (hiking) stops, replace only
those stops with indoor ones (museums, history, food), keep every other day unchanged,
then re-validate.

## Done means
Metrics for N US trips, with and without the validator: pass rate, repairs per trip,
tool calls per trip, average daily km, pricing coverage, estimated share, and the share
of stops kept after a weather replan.
# Voyage Build Plan

## Problem Statement
AI trip planners generate detailed itineraries, but few verify them—wrong addresses, closed businesses, and unrealistic routing go unchecked. Users trust the plan and waste time discovering issues on the ground.

## Scope
- **US only**: Route planning and venue validation for US cities only.
- **$0 budget**: Use free APIs and open-source tools; no paid services.

## Build Order

- [ ] Parse input (City, ST format validation)
- [ ] Integrate location tools (Geoapify, business/venue data)
- [ ] Build trip validator (checks addresses, hours, routes)
- [ ] Agent loop (LLM + tools interaction; Groq backend)
- [ ] Weather-based replanning logic
- [ ] Evaluation suite (correctness, robustness, edge cases)
- [ ] README & user documentation
- [ ] (later) Map and PDF export, built fresh against the Itinerary model

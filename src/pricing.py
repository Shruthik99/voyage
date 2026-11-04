"""What a stop costs, and where that number came from.

The seam between a Place and a Stop. The LLM never supplies prices; this does,
or admits it cannot. SPEC's layered cost model:

1. Known price from a source (NPS entranceFees, OSM charge tag) -- confidence known
2. Known free (a no-fee flag) -- 0 to 0, confidence known
3. Food: the traveler's meal allowance -- confidence estimate
4. Other types: a low-high range from a documented sample of real prices, with the
   method and the sample in docs/PRICING.md -- confidence estimate
5. A traveler-entered price overrides everything above

None of those layers exist yet: docs/PRICING.md has not been written, and neither
Geoapify's places response nor our Place model carries a fee field today. So every
call returns unknown. That is the honest answer, not a placeholder -- an unknown
cost is a first-class outcome in the Stop model (both costs None) and in V2, which
reports "unpriced" rather than claiming a trip fits its budget.

Consequence worth knowing while this file is a stub: with nothing priced, V2 can
warn but can never fail, and SPEC's pricing-coverage and estimated-share metrics
read zero.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.models import CostBasis, CostConfidence, Place

if TYPE_CHECKING:  # avoids a cycle: agent imports pricing, pricing needs the request type
    from src.agent import TripRequest


class CostFields(BaseModel):
    """The cost half of a Stop, kept separate so pricing owns it alone."""

    cost_low_usd: float | None
    cost_high_usd: float | None
    cost_basis: CostBasis
    cost_source: str
    cost_confidence: CostConfidence


UNPRICED = CostFields(
    cost_low_usd=None,
    cost_high_usd=None,
    cost_basis="per_person",
    cost_source="unpriced",
    cost_confidence="unknown",
)


def price_place(place: Place, request: "TripRequest") -> CostFields:
    """What this place costs, or unknown. Never guesses, never returns a placeholder."""
    return UNPRICED

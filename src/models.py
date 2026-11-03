"""Pydantic models for every boundary: tool results, itineraries, validation.

Kept free of network imports so that constructing or checking a model never
touches httpx or .env.

A note on strictness. Fields our own code produces (the cost invariants on
Stop) are validated here, because a violation is our bug. Fields the LLM
produces and a validator rule checks -- name, source, source_id -- are
deliberately permissive: constraining them here would turn a repairable V4
failure into a ValidationError during output parsing, and V4 would be dead code.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

CostBasis = Literal["per_person", "per_group", "per_vehicle"]
CostConfidence = Literal["known", "estimate", "unknown"]
RuleStatus = Literal["pass", "fail", "warn"]


class GeoPoint(BaseModel):
    city: str
    state: str
    lat: float
    lon: float


class Place(BaseModel):
    name: str
    address: str
    state: str | None
    lat: float
    lon: float
    interest: str
    source: str
    source_id: str


class ForecastDay(BaseModel):
    date: date
    rain_chance: float
    high_f: float
    source: str


class Stop(Place):
    """A Place the pricing step has costed.

    cost_source records where the number came from: an NPS entranceFees entry,
    an OSM charge tag, the traveler's meal allowance, a documented sample in
    docs/PRICING.md, or a traveler-entered price.
    """

    cost_low_usd: float | None
    cost_high_usd: float | None
    cost_basis: CostBasis
    cost_source: str
    cost_confidence: CostConfidence

    @model_validator(mode="after")
    def _check_costs(self) -> "Stop":
        low, high = self.cost_low_usd, self.cost_high_usd

        if self.cost_confidence == "unknown":
            if low is not None or high is not None:
                raise ValueError("unknown cost_confidence requires both costs to be None")
            return self

        if low is None or high is None:
            # A half-priced stop would add to S_low but not S_high, letting V2
            # report "within budget" for a trip that is over it.
            raise ValueError(
                f"{self.cost_confidence} cost_confidence requires both "
                "cost_low_usd and cost_high_usd"
            )
        if low < 0 or high < 0:
            raise ValueError("costs cannot be negative")
        if low > high:
            raise ValueError(f"cost_low_usd ({low}) exceeds cost_high_usd ({high})")
        return self


class Day(BaseModel):
    date: date
    city: str  # the full "City, ST" string, matching a ValidationContext key
    stops: list[Stop]
    forecast: ForecastDay | None = None
    warnings: list[str] = Field(default_factory=list)


class Itinerary(BaseModel):
    days: list[Day]
    warnings: list[str] = Field(default_factory=list)
    estimated_share: float = 0.0
    unknown_count: int = 0
    verified: bool = False


class RuleResult(BaseModel):
    """One rule's verdict: a per-rule summary, or a detail naming what failed."""

    rule: str  # "V1" ... "V8"
    code: str  # "over", "state_mismatch", ... -- stable, unlike the message
    status: RuleStatus
    message: str
    summary: bool = False  # the one always-present result for its rule
    day: int | None = None  # 0-based; messages say "day 1" for the same day
    stop: int | None = None
    source_id: str | None = None  # survives the reindexing a repair causes


class ValidationReport(BaseModel):
    results: list[RuleResult]
    warnings: list[str] = Field(default_factory=list)
    estimated_share: float = 0.0
    unknown_count: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """True when no rule failed. A warning never makes a trip not-ok."""
        return not any(r.status == "fail" for r in self.results)


class ValidationContext(BaseModel):
    """Everything validate() needs that the itinerary itself does not carry."""

    city_points: dict[str, GeoPoint] = Field(default_factory=dict)
    tool_results: set[tuple[str, str]] = Field(default_factory=set)
    travelers: int = Field(ge=1)
    budget_usd: float = Field(ge=0)
    interests: list[str] = Field(default_factory=list)

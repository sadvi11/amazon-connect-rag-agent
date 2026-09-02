"""Request and response contracts as Pydantic models.

These are not decoration over a dict. Three things they buy:

  A rejected request never reaches the model. A missing or oversized question
  is a 422 at the boundary, not a Bedrock call that costs money before failing.

  The response shape is enforced rather than hoped for. The abstention fields
  in particular - `grounded`, `escalate` - are what the contact flow branches
  on, and a handler that forgets one produces a bot that silently never
  escalates.

  The OpenAPI schema is generated from these, so the contract a caller reads
  and the contract the server enforces are the same object. A hand-written
  schema is a second source of truth that drifts.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000,
                          description="The customer's message.")
    contact_id: str = Field(min_length=1, max_length=128)
    session_attributes: dict[str, str] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def not_only_whitespace(cls, v: str) -> str:
        """min_length alone accepts "   ", which reaches retrieval, matches
        nothing, and bills a request to find that out."""
        if not v.strip():
            raise ValueError("question cannot be only whitespace")
        return v.strip()

    @field_validator("session_attributes")
    @classmethod
    def values_are_strings(cls, v: dict) -> dict:
        """Amazon Connect attributes are string-valued and it rejects the
        whole SetContactAttributes call otherwise - at runtime, mid-chat."""
        bad = [k for k, val in v.items() if not isinstance(val, str)]
        if bad:
            raise ValueError(f"attribute values must be strings: {bad}")
        return v


class Citation(BaseModel):
    source: str
    score: float = Field(ge=0.0)
    excerpt: str = Field(max_length=300)


class AskResponse(BaseModel):
    answer: str
    grounded: bool = Field(description="False means the bot declined to answer.")
    escalate: bool = Field(description="The contact flow branches on this.")
    reason: str = Field(default="", description="Why it declined, if it did.")
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    tokens: TokenUsage | None = None

    @field_validator("escalate")
    @classmethod
    def escalation_matches_grounding(cls, v: bool, info) -> bool:
        """An ungrounded answer that does not escalate is the exact failure
        this service exists to prevent: the customer gets "I don't know" and
        then stays in a loop with the bot. Caught here rather than in review.
        """
        grounded = info.data.get("grounded")
        if grounded is False and v is False:
            raise ValueError("an ungrounded answer must escalate")
        return v


class TokenUsage(BaseModel):
    prompt: int = Field(ge=0)
    completion: int = Field(ge=0)
    context_passages: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.prompt + self.completion


class HealthResponse(BaseModel):
    status: Literal["ok"]
    chunks_indexed: int
    min_passages: int
    min_relevance: float


AskResponse.model_rebuild()

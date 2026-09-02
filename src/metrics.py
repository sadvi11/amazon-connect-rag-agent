"""Containment, per-turn latency, and cost per contact.

Containment rate is the number a contact centre is actually managed on: the
share of contacts resolved without a human. It is also the number easiest to
game, and gaming it is the default failure mode - a bot that answers everything
confidently reports 100% containment while quietly making customers angry.

So containment here counts a contact as contained only if it ended without
escalation AND every answer given was grounded. An abstention is deliberately
NOT counted as containment: refusing correctly is the right behaviour, but it
is not a resolved contact, and a metric that pretends otherwise hides the gap
in the knowledge base that caused it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# Published rates, ca-central-1, September 2026. Kept here rather than in a
# README so the cost figure the bot reports is the cost the bill reflects.
CHAT_MESSAGE_USD = 0.010          # per message sent or received
LEX_REQUEST_USD = 0.004           # per text request beyond the free tier
BEDROCK_QUERY_USD = 0.0003        # measured in bedrock-rag-app, per query
LAMBDA_INVOKE_USD = 0.0000002     # effectively free at this volume


@dataclass
class Turn:
    intent: str
    grounded: bool
    escalated: bool
    latency_ms: float
    inbound_messages: int = 1
    outbound_messages: int = 1
    called_model: bool = True

    @property
    def cost_usd(self) -> float:
        cost = (self.inbound_messages + self.outbound_messages) * CHAT_MESSAGE_USD
        cost += LEX_REQUEST_USD + LAMBDA_INVOKE_USD
        if self.called_model:
            cost += BEDROCK_QUERY_USD
        return cost


@dataclass
class Contact:
    contact_id: str
    turns: list[Turn] = field(default_factory=list)

    def record(self, turn: Turn) -> None:
        self.turns.append(turn)

    @property
    def escalated(self) -> bool:
        return any(t.escalated for t in self.turns)

    @property
    def contained(self) -> bool:
        """Resolved without a human, and every answer was grounded.

        Both halves are load-bearing. Without the escalation check a handed-off
        contact counts as a success. Without the grounded check, a bot that
        confidently invents answers scores perfectly - which is precisely the
        behaviour this metric is supposed to discourage.
        """
        if not self.turns or self.escalated:
            return False
        return all(t.grounded for t in self.turns)

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    @property
    def p50_latency_ms(self) -> float:
        return _percentile([t.latency_ms for t in self.turns], 50)

    @property
    def p95_latency_ms(self) -> float:
        return _percentile([t.latency_ms for t in self.turns], 95)


def containment_rate(contacts: list[Contact]) -> float:
    if not contacts:
        return 0.0
    return sum(1 for c in contacts if c.contained) / len(contacts)


def escalation_rate(contacts: list[Contact]) -> float:
    if not contacts:
        return 0.0
    return sum(1 for c in contacts if c.escalated) / len(contacts)


def cost_per_contact(contacts: list[Contact]) -> float:
    if not contacts:
        return 0.0
    return sum(c.cost_usd for c in contacts) / len(contacts)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    Deliberately not a mean. Average latency hides the tail, and the tail is
    what a customer waiting on a chat window actually experiences.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


class Stopwatch:
    """Wall-clock timing for one turn."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return False


def summary(contacts: list[Contact]) -> dict:
    latencies = [t.latency_ms for c in contacts for t in c.turns]
    return {
        "contacts": len(contacts),
        "turns": sum(len(c.turns) for c in contacts),
        "containment_rate": round(containment_rate(contacts), 4),
        "escalation_rate": round(escalation_rate(contacts), 4),
        "cost_per_contact_usd": round(cost_per_contact(contacts), 6),
        "total_cost_usd": round(sum(c.cost_usd for c in contacts), 6),
        "p50_turn_latency_ms": round(_percentile(latencies, 50), 1),
        "p95_turn_latency_ms": round(_percentile(latencies, 95), 1),
    }

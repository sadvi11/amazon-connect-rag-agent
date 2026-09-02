"""Containment, cost and latency.

Containment is the number a contact centre is managed on, and the easiest to
game. These tests exist mostly to stop it being gamed.
"""
from src.metrics import Contact, Turn, containment_rate, cost_per_contact, escalation_rate, summary


def _turn(grounded=True, escalated=False, ms=100.0):
    return Turn(intent="AskQuestion", grounded=grounded,
                escalated=escalated, latency_ms=ms)


def test_a_contact_resolved_without_a_human_is_contained():
    c = Contact("c1", [_turn(), _turn()])
    assert c.contained


def test_an_escalated_contact_is_not_contained():
    c = Contact("c1", [_turn(), _turn(escalated=True)])
    assert not c.contained


def test_a_confidently_wrong_bot_does_not_score_as_contained():
    """The failure this metric exists to prevent. A bot that answers
    everything without grounding never escalates, and would otherwise report
    perfect containment while making customers angry."""
    c = Contact("c1", [_turn(grounded=False), _turn(grounded=False)])
    assert not c.escalated
    assert not c.contained, "ungrounded answers counted as containment"


def test_an_empty_contact_is_not_contained():
    assert not Contact("c1", []).contained


def test_containment_and_escalation_are_measured_across_contacts():
    contacts = [
        Contact("a", [_turn()]),
        Contact("b", [_turn(escalated=True)]),
        Contact("c", [_turn()]),
        Contact("d", [_turn(grounded=False)]),
    ]
    assert containment_rate(contacts) == 0.5
    assert escalation_rate(contacts) == 0.25


def test_cost_per_contact_counts_every_turn():
    one = Contact("a", [_turn()])
    two = Contact("b", [_turn(), _turn()])
    assert cost_per_contact([one, two]) > one.cost_usd


def test_a_turn_that_skipped_the_model_costs_less():
    with_model = Turn("AskQuestion", True, False, 10.0, called_model=True)
    without = Turn("TalkToAgent", True, True, 10.0, called_model=False)
    assert without.cost_usd < with_model.cost_usd


def test_latency_is_reported_at_p95_not_as_a_mean():
    """A mean hides the tail, and the tail is what the customer waiting on the
    chat window actually experiences."""
    # 90 fast turns and 10 slow ones. Note the shape: with only a single
    # outlier in 20 samples, nearest-rank p95 correctly returns the fast
    # value - the slow one is above the 95th percentile, not at it. An
    # earlier version of this test asserted otherwise and was simply wrong
    # about the arithmetic.
    turns = [_turn(ms=100.0) for _ in range(90)] + [_turn(ms=5000.0) for _ in range(10)]
    c = Contact("c1", turns)
    assert c.p50_latency_ms == 100.0
    assert c.p95_latency_ms == 5000.0
    assert c.p95_latency_ms > c.p50_latency_ms

    mean = sum(t.latency_ms for t in turns) / len(turns)
    assert c.p95_latency_ms > mean, "a mean would have hidden the slow tail"


def test_summary_has_no_empty_division():
    assert summary([])["containment_rate"] == 0.0

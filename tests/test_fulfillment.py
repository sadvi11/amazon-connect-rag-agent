"""The Lambda Lex actually calls. No AWS, no Lex, no model."""
import json

from src import fulfillment
from tests.conftest import lex_event


def test_a_grounded_answer_closes_the_intent(store, generate):
    out = fulfillment.lambda_handler(
        lex_event("How long do refunds take?"), None,
        store=store, generate=generate)
    assert out["sessionState"]["dialogAction"]["type"] == "Close"
    assert out["sessionState"]["intent"]["state"] == "Fulfilled"
    assert out["sessionState"]["sessionAttributes"]["escalate"] != "true"
    assert "30 days" in out["messages"][0]["content"]


def test_an_ungrounded_answer_sets_the_flag_the_contact_flow_branches_on(generate):
    """Without escalate=true the flow has no way to know the bot gave up, and
    the customer loops with a bot that has already said it cannot help."""
    from tests.conftest import FakeStore
    out = fulfillment.lambda_handler(
        lex_event("Do you sell aeroplane parts?"), None,
        store=FakeStore(docs=[]), generate=generate)
    assert out["sessionState"]["sessionAttributes"]["escalate"] == "true"


def test_asking_for_a_human_escalates_without_calling_the_model(store):
    """A bot that argues about this is the fastest way to lose a customer -
    and there is no reason to pay for a model call to say yes."""
    called = []

    def tracked(query, passages):
        called.append(query)
        return "..."

    out = fulfillment.lambda_handler(
        lex_event("agent please", intent="TalkToAgent"), None,
        store=store, generate=tracked)
    assert out["sessionState"]["sessionAttributes"]["escalate"] == "true"
    assert called == [], "called the model to honour a request for a human"


def test_the_handoff_carries_context_to_the_agent(generate):
    from tests.conftest import FakeStore
    out = fulfillment.lambda_handler(
        lex_event("where is order A1", intent="CheckOrder",
                  slots={"OrderNumber": {"value": {"interpretedValue": "A1"}}}),
        None, store=FakeStore(docs=[]), generate=generate)
    attrs = out["sessionState"]["sessionAttributes"]
    assert "A1" in attrs["capturedSlots"]
    assert attrs["handoffReason"]
    assert "where is order A1" in attrs["recentTranscript"]


def test_every_session_attribute_is_a_string(store, generate):
    """Connect rejects the whole SetContactAttributes call if any value is not
    a string, and it fails at runtime rather than at deploy."""
    out = fulfillment.lambda_handler(
        lex_event("How long do refunds take?"), None,
        store=store, generate=generate)
    for key, value in out["sessionState"]["sessionAttributes"].items():
        assert isinstance(value, str), f"{key} is {type(value).__name__}"


def test_an_unfilled_slot_does_not_crash_the_handler(store, generate):
    """Lex V2 sends null for a slot it has not captured. Reading it naively
    raises TypeError on the first empty slot."""
    out = fulfillment.lambda_handler(
        lex_event("track my order", intent="CheckOrder",
                  slots={"OrderNumber": None}),
        None, store=store, generate=generate)
    assert out["sessionState"]


def test_the_transcript_accumulates_across_turns(store, generate):
    first = fulfillment.lambda_handler(
        lex_event("How long do refunds take?"), None,
        store=store, generate=generate)
    carried = first["sessionState"]["sessionAttributes"]

    second = fulfillment.lambda_handler(
        lex_event("How do I return an item?", attributes=carried), None,
        store=store, generate=generate)
    transcript = json.loads(second["sessionState"]["sessionAttributes"]["transcript"])
    assert len(transcript) >= 4
    assert any("refunds" in t["text"] for t in transcript)


def test_each_turn_reports_its_own_latency_and_cost(store, generate):
    captured = []
    fulfillment.lambda_handler(
        lex_event("How long do refunds take?"), None,
        store=store, generate=generate, emit=captured.append)
    assert len(captured) == 1
    turn = captured[0]
    assert turn.latency_ms >= 0
    assert turn.cost_usd > 0

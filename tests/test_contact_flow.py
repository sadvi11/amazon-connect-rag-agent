"""The Connect flow and Lex bot as data.

A contact flow is JSON, so it can be checked before import rather than by
clicking through the console and discovering a dead end with a customer in it.
"""
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FLOW = json.loads((ROOT / "connect" / "contact-flow.json").read_text())
BOT = json.loads((ROOT / "lex" / "bot.json").read_text())


def _targets(action):
    t = action.get("Transitions", {})
    out = []
    if "NextAction" in t:
        out.append(t["NextAction"])
    out += [c["NextAction"] for c in t.get("Conditions", [])]
    out += [e["NextAction"] for e in t.get("Errors", [])]
    return out


def test_every_transition_points_at_an_action_that_exists():
    """A dangling transition is accepted at import and fails with a live
    customer in the flow."""
    ids = {a["Identifier"] for a in FLOW["Actions"]}
    for action in FLOW["Actions"]:
        for target in _targets(action):
            assert target in ids, \
                f"{action['Identifier']} points at missing action '{target}'"


def test_no_action_is_unreachable():
    reachable = {FLOW["StartAction"]}
    for action in FLOW["Actions"]:
        reachable.update(_targets(action))
    orphans = {a["Identifier"] for a in FLOW["Actions"]} - reachable
    assert not orphans, f"unreachable: {orphans}"


def test_the_flow_always_ends_at_a_disconnect():
    assert any(a["Type"] == "DisconnectParticipant" for a in FLOW["Actions"])


@pytest.mark.parametrize("action", [a for a in FLOW["Actions"]
                                    if a["Type"] not in ("DisconnectParticipant",)])
def test_every_action_handles_its_own_failure(action):
    """The failure mode this prevents is a customer stuck in silence. Any step
    that can fail must route somewhere - to an agent, not to nothing."""
    t = action.get("Transitions", {})
    assert t.get("Errors") or t.get("NextAction"), \
        f"{action['Identifier']} has no path out on failure"


def test_a_lex_failure_routes_to_a_human_rather_than_looping():
    lex = next(a for a in FLOW["Actions"] if a["Type"] == "ConnectParticipantWithLexBot")
    error_targets = {e["NextAction"] for e in lex["Transitions"].get("Errors", [])}
    assert error_targets, "Lex step has no error handling"
    assert error_targets == {"to-agent"}, \
        f"a Lex failure goes to {error_targets}, which is not a human"


def test_the_flow_branches_on_the_attribute_the_lambda_sets():
    """src/fulfillment.py writes escalate=true. If the flow compares anything
    else, escalation silently never happens - and nothing errors."""
    compare = next(a for a in FLOW["Actions"] if a["Type"] == "Compare")
    assert compare["Parameters"]["ComparisonValue"] == "$.Attributes.escalate"
    operands = [c["Condition"]["Operands"] for c in compare["Transitions"]["Conditions"]]
    assert ["true"] in operands


def test_asking_for_a_human_is_an_intent_the_bot_recognises():
    names = {i["intentName"] for i in BOT["intents"]}
    assert "TalkToAgent" in names


def test_there_is_a_fallback_intent():
    """Below the confidence threshold the bot must have somewhere to go, or
    Lex returns its default message and the customer is stuck."""
    assert any(i.get("parentIntentSignature") == "AMAZON.FallbackIntent"
               for i in BOT["intents"])


def test_every_intent_reaches_the_lambda():
    for intent in BOT["intents"]:
        assert intent.get("fulfillmentCodeHook", {}).get("enabled"), \
            f"{intent['intentName']} does not call the fulfilment hook"


def test_the_order_number_slot_is_required():
    """Capturing it in Lex means the Lambda is never asked to parse an order
    number out of prose."""
    order = next(i for i in BOT["intents"] if i["intentName"] == "CheckOrder")
    slot = next(s for s in order["slots"] if s["slotName"] == "OrderNumber")
    assert slot["valueElicitationSetting"]["slotConstraint"] == "Required"

"""Lambda fulfilment for the Lex V2 bot behind an Amazon Connect chat flow.

Lex handles intent classification and slot capture. This handles everything
after that: answering from the knowledge base when it can, escalating when it
cannot, and recording what each turn cost and how long it took.

The response shape is Lex V2's, not Lex V1's - they differ, and sending the
wrong one produces a bot that fails silently with a generic fallback message
rather than an error anyone can see.
"""
from __future__ import annotations

import json

from . import handoff, metrics, rag

ESCALATION_INTENT = "TalkToAgent"
QUESTION_INTENT = "AskQuestion"


def lambda_handler(event, context, store=None, generate=None, emit=None):
    """Entry point. Dependencies are injectable so the whole path is testable
    without AWS, Lex, or a model - see tests/."""
    session = event.get("sessionState", {})
    intent = session.get("intent", {})
    intent_name = intent.get("name", "")
    slots = _flatten_slots(intent.get("slots") or {})
    attributes = session.get("sessionAttributes") or {}
    transcript = json.loads(attributes.get("transcript", "[]"))
    utterance = event.get("inputTranscript", "")
    contact_id = attributes.get("contactId", "unknown")

    transcript.append({"speaker": "customer", "text": utterance})

    with metrics.Stopwatch() as timer:
        if intent_name == ESCALATION_INTENT:
            result = _escalate(contact_id, transcript, slots,
                               reason="customer asked for a human")
            called_model = False
        else:
            answer = rag.answer(utterance, store, generate)
            if answer.should_escalate:
                result = _escalate(contact_id, transcript, slots,
                                   reason=answer.reason or "not grounded",
                                   message=answer.text)
            else:
                result = _fulfilled(answer.text, intent_name, slots, attributes)
            called_model = True

    turn = metrics.Turn(
        intent=intent_name or QUESTION_INTENT,
        grounded=not result["_escalated"],
        escalated=result["_escalated"],
        latency_ms=timer.elapsed_ms,
        called_model=called_model,
    )
    if emit:
        emit(turn)

    transcript.append({"speaker": "bot", "text": result["_message"]})
    result["sessionState"]["sessionAttributes"] = {
        **attributes,
        "contactId": contact_id,
        "transcript": json.dumps(transcript[-handoff.MAX_TRANSCRIPT_TURNS * 2:]),
        "lastTurnLatencyMs": f"{turn.latency_ms:.0f}",
        "lastTurnCostUsd": f"{turn.cost_usd:.6f}",
        **result.pop("_attributes", {}),
    }
    result.pop("_escalated", None)
    result.pop("_message", None)
    return result


def _fulfilled(message: str, intent_name: str, slots: dict, attributes: dict) -> dict:
    return {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {"name": intent_name or QUESTION_INTENT, "state": "Fulfilled"},
        },
        "messages": [{"contentType": "PlainText", "content": message}],
        "_escalated": False,
        "_message": message,
        # Set explicitly rather than left absent. Session attributes persist
        # across turns, so a stale escalate=true from an earlier turn would
        # otherwise route a perfectly good answer to an agent - and the
        # contact flow branches on exactly this value.
        "_attributes": {"escalate": "false"},
    }


def _escalate(contact_id: str, transcript: list, slots: dict, reason: str,
              message: str | None = None) -> dict:
    """Hand off, and set the attributes Connect will route on.

    escalate=true is what the contact flow branches on. Without it the flow
    has no way to know this turn ended in a handoff, and the customer stays
    in a loop with a bot that has already told them it cannot help.
    """
    text = message or handoff_message()
    ctx = handoff.build_context(contact_id, transcript, slots, reason)
    return {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {"name": ESCALATION_INTENT, "state": "Fulfilled"},
        },
        "messages": [{"contentType": "PlainText", "content": text}],
        "_escalated": True,
        "_message": text,
        "_attributes": {"escalate": "true", **ctx},
    }


def handoff_message() -> str:
    return "Let me get you to someone who can help. One moment."


def _flatten_slots(slots: dict) -> dict:
    """Lex V2 nests slot values three levels deep and uses null for unfilled
    slots. Reading them naively raises TypeError on the first empty slot."""
    out = {}
    for name, slot in (slots or {}).items():
        if not slot:
            out[name] = None
            continue
        value = (slot.get("value") or {})
        out[name] = value.get("interpretedValue") or value.get("originalValue")
    return out

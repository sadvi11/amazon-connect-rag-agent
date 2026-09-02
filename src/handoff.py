"""Escalation to a human, carrying the conversation with it.

The reason people hate chatbots is not that the bot failed. It is being asked
to repeat everything to the human afterwards - which tells the customer the
first five minutes were wasted, and hands the agent a cold start.

Amazon Connect passes contact attributes through to the agent workspace, so
the transcript, the captured slots, and the reason for escalation can all
arrive with the contact. This builds that payload.
"""
from __future__ import annotations

import json

# Connect contact attributes are string-valued, and there is a size limit on
# the attribute set. A long chat will exceed it, so the transcript is trimmed
# to the most recent turns - the ones an agent actually needs to pick up.
MAX_TRANSCRIPT_TURNS = 6
MAX_ATTRIBUTE_CHARS = 1024


def build_context(contact_id: str, transcript: list[dict], slots: dict,
                  reason: str) -> dict:
    """Attributes for the agent workspace.

    Every value is a string because Connect requires it. Sending a dict here
    fails at runtime rather than at deploy, which is the worst time to find out.
    """
    recent = transcript[-MAX_TRANSCRIPT_TURNS:]
    rendered = "\n".join(
        f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in recent
    )

    return {
        "handoffReason": _clip(reason, 256),
        "capturedSlots": _clip(json.dumps(slots, separators=(",", ":")), MAX_ATTRIBUTE_CHARS),
        "recentTranscript": _clip(rendered, MAX_ATTRIBUTE_CHARS),
        "turnsBeforeHandoff": str(len(transcript)),
        "botContactId": contact_id,
        # So the agent opens with the right thing instead of "how can I help?"
        "agentBriefing": _clip(_briefing(slots, reason, recent), MAX_ATTRIBUTE_CHARS),
    }


def _briefing(slots: dict, reason: str, recent: list[dict]) -> str:
    known = ", ".join(f"{k}={v}" for k, v in slots.items() if v) or "nothing captured"
    last_customer = next(
        (t["text"] for t in reversed(recent) if t.get("speaker") == "customer"), ""
    )
    return (
        f"Bot could not resolve: {reason}. "
        f"Already captured: {known}. "
        f"Customer's last message: {last_customer}"
    )


def _clip(value: str, limit: int) -> str:
    """Connect rejects the whole SetContactAttributes call if one value is too
    long, so clip here rather than discovering it mid-conversation."""
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"

from src import handoff


def _transcript(n):
    return [{"speaker": "customer" if i % 2 == 0 else "bot", "text": f"line {i}"}
            for i in range(n)]


def test_the_agent_gets_a_briefing_not_just_a_dump():
    ctx = handoff.build_context("c1", _transcript(4),
                                {"OrderNumber": "A1"}, "not grounded")
    assert "A1" in ctx["agentBriefing"]
    assert "not grounded" in ctx["agentBriefing"]


def test_a_long_chat_is_trimmed_to_recent_turns():
    """Connect limits the attribute set, and rejects the whole call if one
    value is too long."""
    ctx = handoff.build_context("c1", _transcript(50), {}, "reason")
    assert len(ctx["recentTranscript"]) <= handoff.MAX_ATTRIBUTE_CHARS
    assert ctx["recentTranscript"].count("\n") < handoff.MAX_TRANSCRIPT_TURNS


def test_an_oversized_slot_payload_is_clipped():
    ctx = handoff.build_context("c1", _transcript(2), {"note": "x" * 5000}, "r")
    assert len(ctx["capturedSlots"]) <= handoff.MAX_ATTRIBUTE_CHARS


def test_every_value_is_a_string():
    ctx = handoff.build_context("c1", _transcript(4), {"OrderNumber": "A1"}, "r")
    for key, value in ctx.items():
        assert isinstance(value, str), f"{key} is {type(value).__name__}"


def test_it_survives_an_empty_conversation():
    ctx = handoff.build_context("c1", [], {}, "customer asked for a human")
    assert ctx["turnsBeforeHandoff"] == "0"
    assert "nothing captured" in ctx["agentBriefing"]

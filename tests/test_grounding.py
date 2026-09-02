"""The bot must refuse rather than guess.

A confident wrong answer about a refund or a deadline, in the brand's voice,
is the expensive failure. "I don't know" is not.
"""
from src import rag
from tests.conftest import FakeStore


def test_it_answers_when_the_knowledge_base_covers_the_question(store, generate):
    result = rag.answer("How long do refunds take?", store, generate)
    assert result.grounded
    assert not result.should_escalate
    assert "30 days" in result.text


def test_it_refuses_when_nothing_relevant_is_retrieved(generate):
    result = rag.answer("Do you sell aeroplane parts?", FakeStore(docs=[]), generate)
    assert not result.grounded
    assert result.should_escalate
    assert result.text == rag.REFUSAL


def test_it_refuses_on_a_single_hit(generate):
    """One passage is not evidence the corpus covers the topic - it can be a
    coincidence of wording. The query below does retrieve that one passage, so
    this exercises the passage-count gate rather than the empty guard."""
    thin = FakeStore(docs=[("Orders ship within 2 business days.", "shipping")])
    result = rag.answer("When do orders ship?", thin, generate)
    assert result.passages, "fixture no longer retrieves anything"
    assert not result.grounded
    assert "passage" in result.reason


def test_the_model_declining_is_treated_as_an_abstention(store):
    """If the model says it doesn't know despite adequate context, that is not
    a contained answer, however fluent it sounds."""
    def declines(query, passages):
        return "I don't know based on the information provided."

    result = rag.answer("How long do refunds take?", store, declines)
    assert not result.grounded
    assert result.should_escalate
    assert "declined" in result.reason


def test_generation_does_not_run_when_we_are_going_to_refuse(generate):
    """Checking after generating means paying for a call we discard - and
    leaves an unusable answer sitting in the process."""
    called = []

    def tracked(query, passages):
        called.append(query)
        return "..."

    rag.answer("something entirely unrelated", FakeStore(docs=[]), tracked)
    assert called == [], "the model was called before the relevance gate"

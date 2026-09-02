"""Justify MIN_RELEVANCE with an eval set rather than a number someone liked.

A threshold picked by feel never gets revisited and cannot be defended in
review. This makes the number falsifiable: move it and a test says what broke.

Precision matters more than recall here, asymmetrically. Refusing a question
the corpus could have answered costs one transfer to a human. Answering one it
could not costs a customer a wrong answer about their money, in the brand's
voice.

Note which questions belong in which list. Questions that retrieve *nothing*
are caught by the empty-retrieval guard and tell you nothing about the
threshold. The ones that matter here retrieve several passages and are refused
anyway, because none of them is a good enough match.
"""
import pytest

from src import rag
from tests.conftest import FakeStore

ANSWERABLE = [
    "How long do refunds take?",
    "How do I return an item?",
    "When do orders ship?",
    "What does policy RTN-14 cover?",
]

# Retrieve nothing at all - caught by the empty guard, not by a threshold.
OFF_TOPIC = [
    "Do you sell aeroplane parts?",
    "Can I pay in Japanese yen?",
]

# The interesting ones. These share vocabulary with the corpus and retrieve
# four passages each, then get refused on relevance.
PARTIAL_MATCH = [
    "What is the refund policy for aeroplane parts?",
    "Can I return a damaged laptop to your Tokyo store?",
]


@pytest.fixture
def store():
    return FakeStore()


def _generate(query, passages):
    return f"Answer: {passages[0].text}"


@pytest.mark.parametrize("question", ANSWERABLE)
def test_questions_the_corpus_covers_are_answered(question, store):
    result = rag.answer(question, store, _generate)
    assert result.grounded, (
        f"refused a question it could answer ({result.reason}); "
        f"MIN_RELEVANCE={rag.MIN_RELEVANCE} may be too high")


@pytest.mark.parametrize("question", OFF_TOPIC)
def test_wholly_unrelated_questions_retrieve_nothing_and_are_refused(question, store):
    result = rag.answer(question, store, _generate)
    assert not result.grounded
    assert result.passages == []


@pytest.mark.parametrize("question", PARTIAL_MATCH)
def test_partial_matches_are_refused_on_relevance(question, store):
    """These retrieve passages - the threshold is what stops them."""
    result = rag.answer(question, store, _generate)
    assert result.passages, "fixture no longer exercises the threshold"
    assert not result.grounded, (
        f"answered on a weak match; MIN_RELEVANCE={rag.MIN_RELEVANCE} is too "
        f"low, and this is the expensive direction to be wrong in")
    assert "relevance" in result.reason


def test_the_relevance_threshold_is_load_bearing(store):
    """Drop it and the partial matches must start leaking through.

    Written after an earlier version of this test showed MIN_RELEVANCE was
    doing nothing at all: the passage-count gate caught everything, and the
    threshold was decorative while still being cited in review.
    """
    original = rag.MIN_RELEVANCE
    rag.MIN_RELEVANCE = 0.0
    try:
        leaked = [q for q in PARTIAL_MATCH
                  if rag.answer(q, store, _generate).grounded]
    finally:
        rag.MIN_RELEVANCE = original
    assert len(leaked) == len(PARTIAL_MATCH), (
        "lowering the threshold changed nothing, so it is not what refuses "
        "these questions")


@pytest.mark.xfail(reason="known limitation of term-overlap reranking, kept "
                          "visible rather than curated out of the eval set",
                   strict=True)
def test_a_shared_keyword_can_still_produce_a_false_positive(store):
    """The corpus says nothing about salaries, but 'policy' matches the two
    RTN-14 passages well enough to score exactly at the threshold.

    This is the ceiling of scoring by term overlap: it measures vocabulary,
    not meaning. A cross-encoder scores the pair (question, passage) and would
    reject this. The rerank() interface is designed for that swap - the fix is
    a different body, not a different threshold, and raising the threshold to
    hide this one case would start refusing real questions.
    """
    result = rag.answer("What policy covers salary disputes?", store, _generate)
    assert not result.grounded

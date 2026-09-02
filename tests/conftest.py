import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from src.rag import Passage

KB = [
    ("Refunds are processed within 30 days of receiving the returned item.", "refunds"),
    ("Refunds are issued to the original payment method once we receive the return.", "refunds"),
    ("To return an item, use the returns portal and print the prepaid label.", "returns"),
    ("Returns are accepted within 60 days of delivery.", "returns"),
    ("Orders ship within 2 business days from our Calgary warehouse.", "shipping"),
    ("Orders placed after 4pm ship the following business day.", "shipping"),
    ("Policy code RTN-14 covers damaged goods on arrival.", "policy"),
    ("Damaged goods reported under policy RTN-14 are replaced at no cost.", "policy"),
]


def _semantic_tokens(text):
    """Words only - identifiers deliberately dropped.

    This is what makes FakeStore a fair stand-in. Real embeddings map an exact
    code like RTN-14 or an order number to whatever it happens to sit near in
    vector space, which is not the same as matching it. A fixture that split
    on whitespace would "find" RTN-14 by dense search and quietly make the
    hybrid-retrieval test meaningless - it would pass whether or not keyword
    search existed at all.
    """
    return {t for t in text.lower().replace(".", "").split()
            if t.isalpha() and len(t) > 2}


class FakeStore:
    """Stands in for pgvector plus a keyword index."""

    def __init__(self, docs=None):
        self.docs = docs if docs is not None else KB

    def dense_search(self, query, k=8):
        terms = _semantic_tokens(query)
        if not terms:
            return []
        out = []
        for text, source in self.docs:
            overlap = len(terms & _semantic_tokens(text))
            if overlap:
                out.append(Passage(text, overlap / len(terms), source))
        return sorted(out, key=lambda p: p.score, reverse=True)[:k]

    def keyword_search(self, query, k=8):
        """Literal substring matching - the half that catches exact codes."""
        out = [Passage(t, 1.0, s) for t, s in self.docs
               if any(tok in t.lower() for tok in query.lower().split()
                      if len(tok) > 3)]
        return out[:k]


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def generate():
    def _generate(query, passages):
        return f"Based on our policy: {passages[0].text}"
    return _generate


def lex_event(utterance, intent="AskQuestion", slots=None, attributes=None):
    return {
        "inputTranscript": utterance,
        "sessionState": {
            "intent": {"name": intent, "slots": slots or {}},
            "sessionAttributes": attributes or {"contactId": "c-1"},
        },
    }

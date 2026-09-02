"""Retrieval and grounded answering, with abstention enforced in code.

The single most damaging failure in a contact centre bot is not "I don't
know" - customers tolerate that. It is a confident wrong answer about a
refund, a policy, or a deadline, delivered in the brand's voice.

So abstention here is not a prompt asking the model to be careful. It is a
threshold checked before generation runs at all, and a check on the generated
text afterwards. A model that has been told to abstain can be talked out of it
by a determined customer; a function that returns early cannot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Below this best-match score, we do not attempt an answer. The value is
# justified by tests/test_threshold.py, which runs a small eval set of
# questions the corpus does and does not cover and asserts this threshold
# separates them. Change the number and that test tells you what it costs.
MIN_RELEVANCE = 0.5

# A second gate. One strong hit can be a coincidence of wording; two related
# passages is weak evidence that the corpus actually covers the topic.
MIN_PASSAGES = 2

REFUSAL = (
    "I don't have that in my knowledge base, so I'd rather not guess. "
    "Let me get you to someone who can help."
)


@dataclass
class Passage:
    text: str
    score: float
    source: str


@dataclass
class Answer:
    text: str
    grounded: bool
    passages: list[Passage] = field(default_factory=list)
    reason: str = ""

    @property
    def should_escalate(self) -> bool:
        return not self.grounded


def retrieve(query: str, store, k: int = 4) -> list[Passage]:
    """Hybrid retrieval: dense vectors for meaning, keyword for exact terms.

    Dense search alone reliably misses things a contact centre is asked about
    constantly - order numbers, policy codes, product SKUs, the literal phrase
    "30 days". Embeddings map those to whatever they are near in vector space,
    which is not the same as matching them.

    Keyword search alone misses paraphrase, which is how customers actually
    write. Neither is sufficient; the fusion below is.
    """
    dense = store.dense_search(query, k=k * 2)
    keyword = store.keyword_search(query, k=k * 2)
    fused = reciprocal_rank_fusion([dense, keyword])
    return rerank(query, fused)[:k]


def reciprocal_rank_fusion(rankings: list[list[Passage]], k: int = 60) -> list[Passage]:
    """Combine rankings by position rather than by score.

    Dense similarity and keyword relevance are not on the same scale and there
    is no honest way to add them. RRF sidesteps that entirely: it only uses
    where a passage placed in each list, so no normalisation fudge is needed.
    k=60 is the value from the original paper; it damps the influence of any
    single list's top result.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Passage] = {}
    for ranking in rankings:
        for position, passage in enumerate(ranking):
            key = passage.text
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
            seen.setdefault(key, passage)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [Passage(text=t, score=s, source=seen[t].source) for t, s in ordered]


def rerank(query: str, passages: list[Passage]) -> list[Passage]:
    """Second-pass scoring on the shortlist only.

    Retrieval optimises for recall - get the right passage into the top twenty.
    Reranking optimises for precision - get it to position one. Running this
    over the whole corpus would be unaffordable; over twenty candidates it is
    cheap, and it is where most of the quality comes from.

    Term overlap stands in for a cross-encoder here so the tests stay offline
    and free. The interface is what matters: swap this body for a Bedrock
    rerank call and nothing else in the pipeline changes.
    """
    terms = _content_tokens(query)
    if not terms:
        return passages
    scored = []
    for p in passages:
        overlap = len(terms & _content_tokens(p.text)) / len(terms)
        scored.append(Passage(text=p.text, score=overlap, source=p.source))
    return sorted(scored, key=lambda p: p.score, reverse=True)


# Filler words carry no retrieval signal but do dilute an overlap score. With
# them included, "How long do refunds take?" scores 0.25 against a passage
# that answers it perfectly, purely because three of the four query words were
# "how", "long" and "take". A cross-encoder handles this implicitly; a term
# overlap scorer has to be told.
_STOPWORDS = frozenset("""
a an and are as at be by can could do does did for from had has have how i if
in into is it its me my of on or our so than that the their them then there
these they this to us was we were what when where which who why will with
would you your take long much many
""".split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOPWORDS}


def answer(query: str, store, generate) -> Answer:
    """Retrieve, decide whether we may answer at all, then generate.

    The order matters. Generating first and checking afterwards means paying
    for a model call we are going to discard, and it means the unusable answer
    exists in the process where somebody can later be tempted to return it.
    """
    passages = retrieve(query, store)

    if not passages:
        # Guarded separately from the count check below. With MIN_PASSAGES at
        # its usual value this branch is unreachable, but the max() further
        # down raises ValueError on an empty sequence - so lowering that
        # constant would turn a refusal into an unhandled exception, which in
        # a Lambda is a 500 to a customer mid-conversation. Found by
        # tests/test_threshold.py while checking the gates independently.
        return Answer(REFUSAL, grounded=False, passages=[],
                      reason="nothing retrieved")

    if len(passages) < MIN_PASSAGES:
        return Answer(REFUSAL, grounded=False, passages=passages,
                      reason=f"only {len(passages)} passage(s) retrieved")

    best = max(p.score for p in passages)
    if best < MIN_RELEVANCE:
        return Answer(REFUSAL, grounded=False, passages=passages,
                      reason=f"best relevance {best:.2f} < {MIN_RELEVANCE}")

    text = generate(query, passages)

    # The model can still refuse even when the context looked adequate, and it
    # is right to. Treat that as an abstention rather than an answer, so the
    # containment metric does not count it as a success.
    if _reads_as_refusal(text):
        return Answer(text, grounded=False, passages=passages,
                      reason="model declined despite adequate context")

    return Answer(text, grounded=True, passages=passages)


def _reads_as_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in (
        "i don't know", "i do not know", "i'm not sure", "cannot determine",
        "no information", "not in the provided", "unable to answer",
    ))

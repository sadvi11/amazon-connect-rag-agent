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

import asyncio
import inspect
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


async def retrieve_async(query: str, store, k: int = 4) -> list[Passage]:
    """The same retrieval, with the two searches running concurrently.

    Dense and keyword search are independent - one hits pgvector, the other a
    text index - so running them in sequence adds their latencies for no
    reason. Concurrently, the cost is the slower of the two rather than the
    sum, which on a p95 dominated by the vector search is most of the saving.

    This matters more here than in a batch pipeline: a customer is watching a
    chat window, and every millisecond is one they spend looking at a typing
    indicator.

    Accepts either a sync or async store so the same code path works against
    a real client and against the test double - the alternative is two
    retrieval implementations that drift.
    """
    dense, keyword = await asyncio.gather(
        _maybe_await(store.dense_search, query, k=k * 2),
        _maybe_await(store.keyword_search, query, k=k * 2),
    )
    fused = reciprocal_rank_fusion([dense, keyword])
    return rerank(query, fused)[:k]


async def _maybe_await(fn, *args, **kwargs):
    """Await an async callable; run a sync one in a worker thread.

    The thread dispatch is the point. Calling a blocking client inline inside
    a coroutine holds the event loop for the whole call, so the two searches
    that asyncio.gather is supposed to overlap run one after the other. Tests
    still pass - the results are identical - and the concurrency simply does
    not happen. An earlier version of this function did exactly that while a
    comment claimed otherwise.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)


def reciprocal_rank_fusion(rankings: list[list[Passage]], k: int = 60) -> list[Passage]:
    """Combine rankings by position rather than by score.

    Dense similarity and keyword relevance are not on the same scale and there
    is no honest way to add them. RRF sidesteps that entirely: it only uses
    where a passage placed in each list, so no normalisation fudge is needed.

        RRFscore(d) = sum over rankings r of  1 / (k + r(d))

    k=60 is the value from Cormack, Clarke and Buettcher (SIGIR 2009), where
    it "was fixed during a pilot investigation and not altered during
    subsequent validation". Worth being precise about: 60 is empirical, not
    derived, and the paper offers no theory for it. Larger k flattens the gap
    between consecutive ranks, so a passage ranked first contributes less
    disproportionately - but that is a property of the formula, not the
    authors' stated reason for the number.

    https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
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


async def answer_async(query: str, store, generate) -> Answer:
    """Async twin of answer(). The gates are identical - only retrieval and
    generation are awaited - so there is one set of rules, not two."""
    passages = await retrieve_async(query, store)
    verdict = _gate(passages)
    if verdict is not None:
        return verdict

    text = generate(query, passages)
    if inspect.isawaitable(text):
        text = await text

    if _reads_as_refusal(text):
        return Answer(text, grounded=False, passages=passages,
                      reason="model declined despite adequate context")
    return Answer(text, grounded=True, passages=passages)


def _gate(passages: list[Passage]) -> Answer | None:
    """The refusal rules, in one place.

    Extracted so the sync and async paths cannot drift apart. Two copies of a
    safety check is one copy that eventually stops matching the other, and the
    one that stops matching is the one nobody is testing.
    """
    if not passages:
        return Answer(REFUSAL, grounded=False, passages=[],
                      reason="nothing retrieved")
    if len(passages) < MIN_PASSAGES:
        return Answer(REFUSAL, grounded=False, passages=passages,
                      reason=f"only {len(passages)} passage(s) retrieved")
    best = max(p.score for p in passages)
    if best < MIN_RELEVANCE:
        return Answer(REFUSAL, grounded=False, passages=passages,
                      reason=f"best relevance {best:.2f} < {MIN_RELEVANCE}")
    return None


def answer(query: str, store, generate) -> Answer:
    """Retrieve, decide whether we may answer at all, then generate.

    The order matters. Generating first and checking afterwards means paying
    for a model call we are going to discard, and it means the unusable answer
    exists in the process where somebody can later be tempted to return it.
    """
    passages = retrieve(query, store)
    verdict = _gate(passages)
    if verdict is not None:
        return verdict

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

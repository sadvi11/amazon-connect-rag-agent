"""Concurrency, and proof that it is real.

A gather() over two blocking calls made inline is indistinguishable from
running them in sequence, except in wall-clock time. So the test measures
wall-clock time.
"""
import asyncio
import time

import pytest

from src import rag
from src.rag import Passage


class SlowStore:
    """Each search sleeps. If they overlap, the pair costs ~one sleep."""

    DELAY = 0.15

    def dense_search(self, query, k=8):
        time.sleep(self.DELAY)
        return [Passage("Refunds are processed within 30 days.", 0.9, "refunds"),
                Passage("Refunds are issued to the original method.", 0.8, "refunds")]

    def keyword_search(self, query, k=8):
        time.sleep(self.DELAY)
        return [Passage("Refunds are processed within 30 days.", 1.0, "refunds")]


@pytest.mark.asyncio
async def test_the_two_searches_actually_overlap():
    store = SlowStore()
    start = time.perf_counter()
    await rag.retrieve_async("how long do refunds take", store)
    elapsed = time.perf_counter() - start

    sequential = SlowStore.DELAY * 2
    assert elapsed < sequential * 0.75, (
        f"took {elapsed:.3f}s against {sequential:.3f}s sequential - the "
        f"searches are not overlapping, so a blocking call is holding the "
        f"event loop")


@pytest.mark.asyncio
async def test_async_and_sync_paths_return_the_same_thing():
    """Two implementations of the same rules is one that eventually drifts."""
    from tests.conftest import FakeStore
    store = FakeStore()

    def generate(q, p):
        return f"Answer: {p[0].text}"

    sync = rag.answer("How long do refunds take?", store, generate)
    async_ = await rag.answer_async("How long do refunds take?", store, generate)
    assert sync.grounded == async_.grounded
    assert sync.text == async_.text
    assert [p.text for p in sync.passages] == [p.text for p in async_.passages]


@pytest.mark.asyncio
async def test_the_gates_apply_on_the_async_path_too():
    from tests.conftest import FakeStore
    result = await rag.answer_async(
        "Do you sell aeroplane parts?", FakeStore(docs=[]), lambda q, p: "x")
    assert not result.grounded


@pytest.mark.asyncio
async def test_an_async_store_is_awaited_not_wrapped():
    class AsyncStore:
        async def dense_search(self, query, k=8):
            await asyncio.sleep(0)
            return [Passage("a", 0.9, "s"), Passage("b", 0.8, "s")]

        async def keyword_search(self, query, k=8):
            await asyncio.sleep(0)
            return [Passage("a", 1.0, "s")]

    passages = await rag.retrieve_async("a", AsyncStore())
    assert passages, "an async store returned nothing"


@pytest.mark.asyncio
async def test_a_model_refusal_is_an_abstention_on_the_async_path_too():
    """The sync path had this test and the async path did not.

    Fault injection found it: the refusal check exists in both functions, the
    injected fault landed in the async copy, and nothing went red. Duplicated
    logic with single-sided tests is exactly how a safety check rots on one
    branch while looking covered.
    """
    from tests.conftest import FakeStore

    async def declines(query, passages):
        return "I don't know based on the information provided."

    result = await rag.answer_async("How long do refunds take?",
                                    FakeStore(), declines)
    assert not result.grounded
    assert result.should_escalate
    assert "declined" in result.reason

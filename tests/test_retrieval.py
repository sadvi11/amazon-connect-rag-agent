"""Hybrid retrieval, fusion and reranking."""
from src import rag
from src.rag import Passage
from tests.conftest import FakeStore


def test_keyword_search_finds_an_exact_code_dense_search_misses():
    """The case that justifies hybrid retrieval at all. Policy codes, SKUs and
    order numbers are exactly what embeddings are worst at."""
    store = FakeStore()
    dense = store.dense_search("RTN-14")
    keyword = store.keyword_search("RTN-14")
    assert not dense, "fixture no longer demonstrates the dense miss"
    assert any("RTN-14" in p.text for p in keyword)

    fused = rag.reciprocal_rank_fusion([dense, keyword])
    assert any("RTN-14" in p.text for p in fused), \
        "hybrid retrieval lost the exact-match hit"


def test_fusion_ranks_a_passage_found_by_both_above_one_found_by_either():
    both = Passage("in both lists", 0.9, "a")
    dense_only = Passage("dense only", 0.99, "b")
    keyword_only = Passage("keyword only", 0.99, "c")

    fused = rag.reciprocal_rank_fusion(
        [[dense_only, both], [keyword_only, both]])
    assert fused[0].text == "in both lists"


def test_fusion_uses_rank_not_score():
    """Dense similarity and keyword relevance are not on the same scale. If
    fusion added raw scores, the 0.99 would win despite ranking lower in both
    lists."""
    low_score_top_rank = Passage("top of both", 0.01, "a")
    high_score_low_rank = Passage("bottom", 0.99, "b")
    fused = rag.reciprocal_rank_fusion(
        [[low_score_top_rank, high_score_low_rank],
         [low_score_top_rank, high_score_low_rank]])
    assert fused[0].text == "top of both"


def test_reranking_promotes_the_passage_that_matches_the_question():
    passages = [
        Passage("Orders ship within 2 business days.", 0.5, "shipping"),
        Passage("Refunds are processed within 30 days.", 0.5, "refunds"),
    ]
    reranked = rag.rerank("how long do refunds take", passages)
    assert "Refunds" in reranked[0].text


def test_reranking_is_stable_on_an_empty_query():
    passages = [Passage("a", 0.1, "x"), Passage("b", 0.2, "y")]
    assert rag.rerank("", passages) == passages

"""Context packing and token budget."""
from src.prompts import CONTEXT_TOKEN_BUDGET, build_prompt, estimate_tokens, pack_context
from src.rag import Passage


def _passages(n, size=200):
    return [Passage(f"passage {i} " + "word " * size, 1.0 - i * 0.05, f"s{i}")
            for i in range(n)]


def test_packing_stops_at_the_budget():
    packed = pack_context(_passages(20), budget=300)
    assert packed.estimated_tokens <= 300
    assert packed.dropped > 0


def test_it_reports_what_it_dropped():
    """Silent truncation reads as 'we used everything' when it did not."""
    packed = pack_context(_passages(20), budget=200)
    assert packed.dropped == 20 - len(packed.used)


def test_the_strongest_passage_is_first_and_the_second_is_last():
    """Models attend most reliably to the edges. Sorting purely by score
    descending buries the second-best passage in the weakest position."""
    packed = pack_context(_passages(5, size=5), budget=CONTEXT_TOKEN_BUDGET)
    assert len(packed.used) == 5
    scores = [p.score for p in packed.used]
    assert scores[0] == max(scores)
    assert scores[-1] == sorted(scores, reverse=True)[1]


def test_a_short_list_is_left_alone():
    passages = _passages(2, size=5)
    assert pack_context(passages).used == passages


def test_the_prompt_forbids_outside_knowledge_and_names_the_refusal():
    prompt, _ = build_prompt("q", _passages(2, size=5))
    assert "only from the context" in prompt
    assert "I don't know" in prompt
    assert "Do not guess" in prompt


def test_the_prompt_carries_source_labels_so_answers_can_cite():
    prompt, _ = build_prompt("q", _passages(2, size=5))
    assert "[s0]" in prompt


def test_the_estimate_is_within_a_usable_margin():
    """It only has to be good enough to decide whether one more passage fits."""
    text = "This is a sentence of ordinary English prose. " * 20
    assert 0.6 <= estimate_tokens(text) / (len(text.split()) * 1.3) <= 1.6

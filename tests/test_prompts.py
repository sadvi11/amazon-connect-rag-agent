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


# ── The few-shot scaffold ────────────────────────────────────────────────────

def test_the_examples_appear_before_the_real_context():
    """Order matters: examples establish the pattern, then the live question."""
    from src import prompts
    prompts.USE_FEW_SHOT = True
    prompt, _ = build_prompt("How long do refunds take?", _passages(2, size=5))
    assert prompt.index("Examples of the expected answer format") < prompt.index("Context:\n")


def test_one_example_demonstrates_a_refusal():
    """The point of the scaffold. An instruction to refuse is a sentence the
    model weighs against being helpful; a demonstrated refusal is a pattern it
    copies."""
    from src import prompts
    assert any(answer.strip() == "I don't know." for _, _, answer in prompts.FEW_SHOT)


def test_every_example_answer_cites_or_refuses():
    """A worked example that states a fact without a citation teaches the model
    to do the same."""
    from src import prompts
    for _, _, answer in prompts.FEW_SHOT:
        assert "[" in answer or answer.strip() == "I don't know."


def test_the_scaffold_can_be_turned_off():
    """So the two versions can be compared rather than argued about."""
    from src import prompts
    prompts.USE_FEW_SHOT = False
    off, _ = build_prompt("q", _passages(2, size=5))
    prompts.USE_FEW_SHOT = True
    on, _ = build_prompt("q", _passages(2, size=5))
    assert len(on) > len(off)
    assert "Examples of the expected" not in off


def test_the_examples_are_counted_against_the_budget():
    """They are tokens on every turn. If they were free the budget would be
    lying."""
    from src import prompts
    prompts.USE_FEW_SHOT = True
    assert prompts.estimate_tokens(prompts.render_examples()) > 50

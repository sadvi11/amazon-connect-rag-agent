"""Prompt construction and context packing under a token budget.

Two ideas here that are easy to get wrong.

**Packing is not "send the top k".** Every passage costs tokens, and tokens
cost money and latency on every single turn. Past a point, more context makes
answers worse rather than better - the relevant passage is diluted by three
that merely mention the same words. So passages are added in rank order until
the budget is spent, and the budget is deliberately smaller than the model's
context window.

**Ordering matters within the packed context.** Models attend most reliably to
the beginning and end of their input, so the strongest passage goes first and
the second-strongest goes last, with the weaker ones in the middle. That is
the opposite of the obvious "sort by score descending" and costs nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

# Deliberately below the model's window. The limit that matters is the one
# where cost and latency stop being worth the marginal passage, not the one
# where the API starts returning errors.
CONTEXT_TOKEN_BUDGET = 1200
CHARS_PER_TOKEN = 4  # rough for English; see estimate_tokens

SYSTEM = """You are a support assistant. Answer only from the context provided.

If the context does not contain the answer, say exactly: I don't know.
Do not use outside knowledge. Do not guess. Do not apologise at length.
Cite the source name in brackets after any fact you state."""

# Two worked examples, shown to the model before the real question.
#
# These are not chain-of-thought. This is a support bot: the reasoning is one
# hop, and asking it to think step by step would spend tokens narrating a
# lookup. What the examples teach is *format and refusal behaviour* - cite the
# source, and decline when the context does not cover it.
#
# The second example is the one that matters. An instruction to refuse is a
# sentence the model can weigh against being helpful; a demonstrated refusal is
# a pattern it can copy. Instruction-only prompting reliably under-refuses,
# because helpfulness is what the model was trained toward.
FEW_SHOT = [
    (
        "How long do refunds take?",
        "[refunds]\nRefunds are processed within 30 days of receiving the returned item.",
        "Refunds are processed within 30 days of receiving the returned item [refunds].",
    ),
    (
        "Can I pay in Japanese yen?",
        "[refunds]\nRefunds are issued to the original payment method.\n\n"
        "[shipping]\nOrders ship within 2 business days.",
        "I don't know.",
    ),
]


def render_examples() -> str:
    """The few-shot block, or empty when disabled.

    Kept separate from build_prompt so the cost of the examples can be measured
    on its own - see scripts/measure_few_shot.py.
    """
    if not USE_FEW_SHOT:
        return ""
    blocks = []
    for question, context, answer in FEW_SHOT:
        blocks.append(f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}")
    return "Examples of the expected answer format:\n\n" + "\n\n---\n\n".join(blocks) + "\n\n---\n\n"


# Toggle so the two prompt versions can be compared rather than argued about.
USE_FEW_SHOT = True


@dataclass
class PackedContext:
    text: str
    used: list
    dropped: int
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    """Character-based estimate, deliberately not a tokeniser.

    A real tokeniser is a dependency and a model-specific one. For budgeting
    - deciding whether one more passage fits - an estimate that is consistently
    within about 15% is enough, and being wrong in the safe direction costs a
    passage rather than a failed request. The number reported as *actual* usage
    comes back from the model response, not from here.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def pack_context(passages: list, budget: int = CONTEXT_TOKEN_BUDGET) -> PackedContext:
    """Fill the budget in rank order, then reorder for attention."""
    kept, used_tokens = [], 0
    for passage in passages:
        cost = estimate_tokens(passage.text) + 8  # separator and source label
        if used_tokens + cost > budget:
            break
        kept.append(passage)
        used_tokens += cost

    ordered = _strongest_first_and_last(kept)
    rendered = "\n\n".join(
        f"[{p.source}]\n{p.text}" for p in ordered
    )
    return PackedContext(text=rendered, used=ordered,
                         dropped=len(passages) - len(kept),
                         estimated_tokens=estimate_tokens(rendered))


def _strongest_first_and_last(passages: list) -> list:
    """Best passage at the start, second best at the end, rest in the middle.

    Models attend most reliably to the edges of their input. Sorting purely by
    score descending buries the second-best passage in the middle, which is
    the weakest position in the window.
    """
    if len(passages) < 3:
        return passages
    ranked = sorted(passages, key=lambda p: p.score, reverse=True)
    first, second, *rest = ranked
    return [first, *rest, second]


def build_prompt(question: str, passages: list,
                 budget: int = CONTEXT_TOKEN_BUDGET) -> tuple[str, PackedContext]:
    packed = pack_context(passages, budget)
    prompt = (
        f"{SYSTEM}\n\n"
        f"{render_examples()}"
        f"Context:\n{packed.text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt, packed

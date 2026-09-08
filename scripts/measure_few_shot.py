#!/usr/bin/env python3
"""What do the few-shot examples cost?

Measured rather than asserted, and measurable without a model call: the
examples are a fixed string prepended to every prompt, so their token cost and
their effect on the context budget are both deterministic.

What this does NOT measure is whether they improve answers. That needs real
generations against the eval set, and it costs money. Stated here so the number
below is not mistaken for a quality result.

    python3 scripts/measure_few_shot.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import prompts
from src.rag import Passage

CORPUS = [
    Passage("Refunds are processed within 30 days of receiving the returned item.", 1.0, "refunds"),
    Passage("Refunds are issued to the original payment method once we receive the return.", .9, "refunds"),
    Passage("To return an item, use the returns portal and print the prepaid label.", .8, "returns"),
    Passage("Orders ship within 2 business days from our Calgary warehouse.", .7, "shipping"),
    Passage("Policy code RTN-14 covers damaged goods on arrival.", .6, "policy"),
]
QUESTION = "How long do refunds take?"


def measure(enabled: bool) -> dict:
    prompts.USE_FEW_SHOT = enabled
    prompt, packed = prompts.build_prompt(QUESTION, CORPUS)
    return {
        "prompt_tokens": prompts.estimate_tokens(prompt),
        "examples_tokens": prompts.estimate_tokens(prompts.render_examples()),
        "context_tokens": packed.estimated_tokens,
        "passages_kept": len(packed.used),
        "passages_dropped": packed.dropped,
    }


def main() -> int:
    off = measure(False)
    on = measure(True)
    prompts.USE_FEW_SHOT = True

    print(f"  {'':22} {'zero-shot':>12} {'few-shot':>12} {'delta':>10}")
    print("  " + "-" * 60)
    for k in off:
        d = on[k] - off[k]
        print(f"  {k:22} {off[k]:>12} {on[k]:>12} {d:>+10}")

    overhead = on["examples_tokens"]
    pct = overhead / off["prompt_tokens"] * 100
    print(f"\n  The examples cost {overhead} tokens - {pct:.0f}% on top of the "
          f"zero-shot prompt,")
    print(f"  paid on every single turn.")

    # The budget is fixed, so the examples compete with retrieved passages only
    # once the context is full. On a small corpus they are free in that sense.
    if on["passages_kept"] < off["passages_kept"]:
        print(f"  They also cost {off['passages_kept'] - on['passages_kept']} "
              f"retrieved passage(s) at the current budget.")
    else:
        print(f"  At this corpus size they cost no retrieved passages - the "
              f"context budget\n  is not the binding constraint yet.")

    print("\n  Not measured: whether answers improve. That needs real "
          "generations against\n  the eval set, and it costs money.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

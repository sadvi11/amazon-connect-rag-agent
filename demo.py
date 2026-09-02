#!/usr/bin/env python3
"""Run a set of conversations through the whole path and report the numbers.

No AWS account, no Lex, no model, no cost. The store and the generator are
substituted; everything between them - intent routing, hybrid retrieval,
fusion, reranking, the abstention gates, escalation and the handoff payload -
is the code that would run in Lambda.

    python demo.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from src import fulfillment, metrics
from tests.conftest import FakeStore, lex_event

CONVERSATIONS = [
    ("covered by the knowledge base", [
        ("How long do refunds take?", "AskQuestion"),
        ("How do I return an item?", "AskQuestion"),
    ]),
    ("partially matching, must not be guessed at", [
        ("What is the refund policy for aeroplane parts?", "AskQuestion"),
    ]),
    ("wholly outside the corpus", [
        ("Do you sell aeroplane parts?", "AskQuestion"),
    ]),
    ("customer asks for a human immediately", [
        ("I want to speak to an agent", "TalkToAgent"),
    ]),
    ("mixed: answered, then escalated", [
        ("When do orders ship?", "AskQuestion"),
        ("Can I pay in Japanese yen?", "AskQuestion"),
    ]),
]


def generate(query, passages):
    return f"Based on our help centre: {passages[0].text}"


def main() -> int:
    store = FakeStore()
    contacts = []

    for label, turns in CONVERSATIONS:
        contact = metrics.Contact(label)
        attributes = {"contactId": label}
        print(f"\n--- {label} ---")
        for utterance, intent in turns:
            response = fulfillment.lambda_handler(
                lex_event(utterance, intent=intent, attributes=attributes),
                None, store=store, generate=generate,
                emit=contact.record)
            attributes = response["sessionState"]["sessionAttributes"]
            reply = response["messages"][0]["content"]
            marker = "->" if attributes.get("escalate") != "true" else "!!"
            print(f"  customer: {utterance}")
            print(f"  bot   {marker} {reply}")
            if attributes.get("escalate") == "true":
                print(f"  handoff  : {attributes.get('agentBriefing', '')[:100]}")
        contacts.append(contact)

    print("\n" + "=" * 62)
    print(json.dumps(metrics.summary(contacts), indent=2))
    print("=" * 62)
    print("No AWS call was made and nothing was billed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

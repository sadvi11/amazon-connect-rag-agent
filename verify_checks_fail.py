#!/usr/bin/env python3
"""Reintroduce each real defect and confirm a test catches it.

Every entry was either a bug found while building this or a control the whole
design depends on. A green suite proves nothing until it has been shown to go
red - and in a contact centre bot the dangerous failures are all silent ones:
a Compare that never matches, an attribute never set, a threshold that refuses
nothing. None of them raise.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent

FAULTS = [
    ("src/rag.py",
     "MIN_RELEVANCE = 0.5",
     "MIN_RELEVANCE = 0.0",
     "the bot answers weak matches instead of refusing"),

    ("src/rag.py",
     "MIN_PASSAGES = 2",
     "MIN_PASSAGES = 1",
     "one coincidental hit counts as coverage"),

    ("src/rag.py",
     '''    if _reads_as_refusal(text):
        return Answer(text, grounded=False, passages=passages,
                      reason="model declined despite adequate context")''',
     '''    if False:
        return Answer(text, grounded=False, passages=passages,
                      reason="model declined despite adequate context")''',
     "a model saying 'I don't know' counts as a contained answer"),

    ("src/fulfillment.py",
     '"_attributes": {"escalate": "true", **ctx},',
     '"_attributes": {**ctx},',
     "escalation never sets the flag the contact flow branches on"),

    ("src/fulfillment.py",
     '"_attributes": {"escalate": "false"},',
     '"_attributes": {},',
     "a stale escalate=true persists into a successful turn"),

    ("src/metrics.py",
     "        return all(t.grounded for t in self.turns)",
     "        return True",
     "a confidently wrong bot reports perfect containment"),

    ("src/handoff.py",
     "    recent = transcript[-MAX_TRANSCRIPT_TURNS:]",
     "    recent = transcript",
     "an oversized transcript is sent to Connect and rejected whole"),

    ("connect/contact-flow.json",
     '"ComparisonValue": "$.Attributes.escalate"',
     '"ComparisonValue": "$.Attributes.escalated"',
     "the flow compares an attribute nothing ever sets"),

    ("connect/contact-flow.json",
     '''            "NextAction": "to-agent",
            "ErrorType": "NoMatchingCondition"''',
     '''            "NextAction": "ask-bot",
            "ErrorType": "NoMatchingCondition"''',
     "a Lex failure loops the customer instead of finding a human"),
]


def tests_pass() -> bool:
    return subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"],
                          cwd=ROOT, capture_output=True).returncode == 0


def main() -> int:
    if not tests_pass():
        print("::error::the suite is red before any fault was injected")
        return 1
    print("baseline: suite green\n")

    failures = 0
    for filename, good, bad, description in FAULTS:
        path = ROOT / filename
        original = path.read_text()
        if good not in original:
            print(f"::error::{filename}: anchor not found, update FAULTS:\n{good[:90]}")
            return 1
        path.write_text(original.replace(good, bad, 1))
        try:
            still_green = tests_pass()
        finally:
            path.write_text(original)

        if still_green:
            print(f"FAIL  nothing caught it: {description}")
            failures += 1
        else:
            print(f"ok    caught: {description}")

    print()
    if failures:
        print(f"{failures} defect(s) can be reintroduced without failing anything.")
        return 1
    print(f"All {len(FAULTS)} defects are caught. The checks are load-bearing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

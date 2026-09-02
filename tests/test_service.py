"""The FastAPI surface."""
import pytest
from fastapi.testclient import TestClient

from src import service
from tests.conftest import FakeStore


@pytest.fixture
def client():
    service.STORE = FakeStore()
    service.GENERATE = lambda q, p: f"Based on our help centre: {p[0].text}"
    service.CONTACTS.clear()
    return TestClient(service.app)


def test_health_reports_the_thresholds_actually_in_force(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["min_passages"] == service.rag.MIN_PASSAGES
    assert body["min_relevance"] == service.rag.MIN_RELEVANCE


def test_a_covered_question_is_answered_with_citations(client):
    body = client.post("/ask", json={"question": "How long do refunds take?",
                                     "contact_id": "c1"}).json()
    assert body["grounded"] is True
    assert body["escalate"] is False
    assert body["citations"]
    assert body["tokens"]["prompt"] > 0


def test_an_uncovered_question_refuses_and_escalates(client):
    body = client.post("/ask", json={"question": "Do you sell aeroplane parts?",
                                     "contact_id": "c1"}).json()
    assert body["grounded"] is False
    assert body["escalate"] is True
    assert body["reason"]


def test_a_whitespace_question_is_rejected_before_any_model_call(client):
    """422 at the boundary rather than a billed request that finds nothing."""
    assert client.post("/ask", json={"question": "   ",
                                     "contact_id": "c1"}).status_code == 422


def test_a_missing_contact_id_is_rejected(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_an_oversized_question_is_rejected(client):
    assert client.post("/ask", json={"question": "x" * 5000,
                                     "contact_id": "c1"}).status_code == 422


def test_every_response_carries_a_request_id_and_latency(client):
    r = client.post("/ask", json={"question": "How long do refunds take?",
                                  "contact_id": "c1"})
    assert r.headers["x-request-id"]
    assert float(r.headers["x-latency-ms"]) >= 0


def test_a_supplied_request_id_is_echoed_back(client):
    r = client.post("/ask", headers={"x-request-id": "trace-me"},
                    json={"question": "hi", "contact_id": "c1"})
    assert r.headers["x-request-id"] == "trace-me"


def test_metrics_aggregate_across_contacts(client):
    client.post("/ask", json={"question": "How long do refunds take?",
                              "contact_id": "a"})
    client.post("/ask", json={"question": "Do you sell aeroplane parts?",
                              "contact_id": "b"})
    body = client.get("/metrics").json()
    assert body["contacts"] == 2
    assert body["containment_rate"] == 0.5


def test_the_openapi_schema_is_generated(client):
    """The contract a caller reads and the one the server enforces are the
    same object, rather than a hand-written schema that drifts."""
    schema = client.get("/openapi.json").json()
    assert "AskRequest" in schema["components"]["schemas"]
    assert "/ask" in schema["paths"]


# ── The response contract itself ─────────────────────────────────────────────
#
# These construct the model directly. Going through /ask cannot exercise the
# validator, because the handler derives escalate from grounded and so never
# produces the invalid combination - which meant the validator was untested
# while looking covered. Fault injection found that.

def test_an_ungrounded_answer_cannot_be_returned_without_escalating():
    """The failure this prevents: the customer is told "I don't know" and then
    left in a loop with the bot, because nothing set the escalation flag."""
    import pytest as _pytest
    from pydantic import ValidationError

    from src.schemas import AskResponse

    with _pytest.raises(ValidationError):
        AskResponse(answer="I don't know", grounded=False, escalate=False,
                    latency_ms=1.0, cost_usd=0.01)


def test_a_grounded_answer_may_decline_to_escalate():
    from src.schemas import AskResponse
    ok = AskResponse(answer="30 days", grounded=True, escalate=False,
                     latency_ms=1.0, cost_usd=0.01)
    assert ok.escalate is False


def test_a_grounded_answer_may_still_escalate():
    """A customer asking for a human after a good answer is not a failure."""
    from src.schemas import AskResponse
    ok = AskResponse(answer="30 days", grounded=True, escalate=True,
                     latency_ms=1.0, cost_usd=0.01)
    assert ok.escalate is True

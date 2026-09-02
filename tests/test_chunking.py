"""Chunking sets the ceiling on retrieval quality."""
from src.chunking import TARGET_CHARS, chunk_document

DOC = """# Refunds

Refunds are processed within 30 days of receiving the returned item. A refund
is issued to the original payment method once we receive the return.

We cannot refund items marked final sale. Gift purchases are refunded as store
credit unless the original receipt is provided at the time of return.

# Returns

To return an item, use the returns portal and print the prepaid label.
"""


def test_a_short_section_is_not_absorbed_into_the_previous_one():
    """The bug this exists for: the Returns section is under the minimum chunk
    size, and an earlier version merged it backwards into a Refunds chunk. It
    then embedded under the wrong heading and was unreachable by anyone asking
    about returns - silently, with no error anywhere."""
    chunks = chunk_document(DOC, "help.md")
    headings = {c.heading for c in chunks}
    assert "Returns" in headings, "the Returns section was absorbed"
    returns = [c for c in chunks if c.heading == "Returns"]
    assert "returns portal" in returns[0].text


def test_the_heading_travels_with_the_chunk():
    """Retrieval sees only the chunk text. "within 30 days" is useless without
    "Refunds", and nobody scrolls up inside a vector index.

    The document below is deliberately written so the chunk body does NOT
    begin with the heading word. An earlier version of this test used a body
    starting with "Refunds are processed..." and asserted
    startswith("Refunds") - which passed whether or not the heading was
    prepended at all. Fault injection caught that it was testing nothing.
    """
    doc = "# Refunds\n\nProcessing takes 30 days from the date we receive the item back.\n"
    chunk = chunk_document(doc, "help.md")[0]
    assert chunk.heading == "Refunds"
    assert not chunk.text.startswith("Refunds"), "fixture no longer isolates the heading"
    assert chunk.embedding_text.startswith("Refunds")
    assert chunk.embedding_text != chunk.text


def test_chunks_respect_the_target_size():
    long_doc = "# T\n\n" + ("This is a sentence about refunds. " * 200)
    for chunk in chunk_document(long_doc, "x.md"):
        assert len(chunk.text) <= TARGET_CHARS * 1.5


def test_long_text_is_split_on_sentences_not_mid_word():
    long_doc = "# T\n\n" + ("Refunds take thirty days to process. " * 100)
    for chunk in chunk_document(long_doc, "x.md"):
        assert not chunk.text.endswith(" Refund")
        assert chunk.text.strip()[-1] in ".!?" or len(chunk.text) < 50


def test_consecutive_chunks_overlap():
    """A sentence near a boundary belongs to both neighbours. Without overlap,
    an answer sitting across the seam is retrievable from neither side."""
    long_doc = "# T\n\n" + " ".join(
        f"Sentence number {i} about refunds and returns." for i in range(80))
    chunks = chunk_document(long_doc, "x.md")
    assert len(chunks) > 1
    first_tail = set(chunks[0].text.split()[-12:])
    second_head = set(chunks[1].text.split()[:12])
    assert first_tail & second_head, "no overlap between consecutive chunks"


def test_a_document_with_no_headings_still_chunks():
    chunks = chunk_document("Just prose. " * 100, "flat.txt")
    assert chunks
    assert all(c.source == "flat.txt" for c in chunks)


def test_chunks_are_indexed_in_order():
    chunks = chunk_document(DOC, "help.md")
    assert [c.index for c in chunks] == list(range(len(chunks)))

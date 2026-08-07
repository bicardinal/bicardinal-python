"""Live tests against the real provider APIs.

These cost real money, so they are opt-in twice over: the provider keys must be
present *and* BICARDINAL_LIVE_TESTS must be set.

    BICARDINAL_LIVE_TESTS=1 pytest tests/test_live.py

Every payload here is deliberately tiny. The whole module bills well under one
cent. Keep it that way: these exist to check that our pricing assumptions still
match what the APIs report, not to exercise throughput.
"""

from __future__ import annotations

import os
import struct
import uuid
import zlib
from decimal import Decimal

import pytest

from bicardinal import Bicardinal
from bicardinal import Config
from bicardinal.office.pricing import LONG_CONTEXT_INPUT_THRESHOLD
from bicardinal.office.pricing import STANDARD
from bicardinal.office.pricing import token_usage
from bicardinal.office.types import TokenKey
from bicardinal.office.types import Usage

live = pytest.mark.skipif(
    not os.environ.get("BICARDINAL_LIVE_TESTS")
    or not os.environ.get("OPENAI_API_KEY"),
    reason="set BICARDINAL_LIVE_TESTS=1 and provider keys to run live tests",
)
needs_mistral = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"), reason="needs MISTRAL_API_KEY"
)

LUNA = "gpt-5.6-luna"


def tiny_png(n: int = 8) -> bytes:
    """An n x n solid PNG, handmade so the test needs no image library."""
    raw = b"".join(b"\x00" + bytes([200, 30, 30] * n) for _ in range(n))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def tiny_pdf(text: bytes = b"Revenue grew 12 percent") -> bytes:
    """A one-page PDF carrying a single line of text."""
    stream = b"BT /F1 24 Tf 72 700 Td (" + text + b") Tj ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(pdf))
        pdf += str(i).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    start = len(pdf)
    pdf += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()
    return (
        pdf
        + b"trailer\n<< /Size "
        + str(len(objs) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(start).encode()
        + b"\n%%EOF\n"
    )


@pytest.fixture
def store(tmp_path):
    return Bicardinal(tmp_path / "data", config=Config(chunk_size=128))


@pytest.fixture
def openai_client():
    from openai import OpenAI

    return OpenAI()


# -- the response shape our pricing reads ----------------------------------


@live
def test_response_carries_the_fields_pricing_depends_on(openai_client):
    resp = openai_client.responses.create(
        model=LUNA,
        reasoning={"effort": "low"},
        instructions="Reply with exactly one word.",
        input="CONTEXT: cats purr",
    )
    details = resp.usage.input_tokens_details
    assert hasattr(details, "cached_tokens")
    assert hasattr(details, "cache_write_tokens")
    assert hasattr(resp.usage.output_tokens_details, "reasoning_tokens")

    # Cache reads and writes are slices of input_tokens, never additions.
    assert details.cached_tokens + details.cache_write_tokens <= resp.usage.input_tokens

    # A standard request reports "default"; we must bill it at standard rates.
    usage = token_usage(resp, operation="summarize", model=LUNA)
    key = next(iter(usage.tokens))
    assert key.service_tier == STANDARD
    assert key.model == LUNA
    assert not key.long_context
    assert resp.usage.input_tokens < LONG_CONTEXT_INPUT_THRESHOLD

    cost = usage.cost()
    assert cost.is_complete
    assert cost.total_usd > 0


@live
def test_flex_tier_is_reported_and_priced_as_flex(openai_client):
    resp = openai_client.responses.create(
        model=LUNA,
        reasoning={"effort": "low"},
        instructions="Reply with exactly one word.",
        input="CONTEXT: dogs bark",
        service_tier="flex",
    )
    usage = token_usage(resp, operation="summarize", model=LUNA, service_tier="flex")
    key = next(iter(usage.tokens))
    assert key.service_tier == "flex"

    # The very same counts billed at standard rates cost twice as much. Build
    # the comparison from the tally, not by re-reading the response: the
    # response says "flex", and the tier it reports is the one that wins.
    tally = usage.tokens[key]
    standard = Usage()
    standard.add_tokens(
        TokenKey(operation="summarize", model=LUNA, service_tier=STANDARD),
        input_tokens=tally.input_tokens,
        cached_input_tokens=tally.cached_input_tokens,
        cache_write_tokens=tally.cache_write_tokens,
        output_tokens=tally.output_tokens,
    )
    assert usage.cost().total_usd * 2 == standard.cost().total_usd


@live
def test_cache_write_then_read_is_priced_from_real_counters(openai_client):
    """A cold call writes the prefix, a warm call reads it, at different rates."""
    # Unique per run, or a previous run's cache would make the "cold" call warm.
    nonce = uuid.uuid4().hex
    prefix = f"Report {nonce}. " + (
        "The quarterly report describes revenue, margin and headcount. " * 130
    )

    def call():
        return openai_client.responses.create(
            model=LUNA,
            reasoning={"effort": "low"},
            instructions=prefix,
            input="Reply with the single word OK.",
        )

    cold = token_usage(call(), operation="summarize", model=LUNA)
    warm = token_usage(call(), operation="summarize", model=LUNA)

    cold_tally = next(iter(cold.tokens.values()))
    warm_tally = next(iter(warm.tokens.values()))
    assert cold_tally.cache_write_tokens > 1_000  # over the 1024-token minimum
    assert warm_tally.cached_input_tokens > 1_000

    # The write carries a 1.25x surcharge, the read a 90% discount, so the
    # warm call must come out materially cheaper than the cold one.
    assert warm.cost().total_usd < cold.cost().total_usd


# -- whole-pipeline ingestion ----------------------------------------------


@live
def test_text_ingestion_prices_end_to_end(store):
    col = store.create("live_text")
    col.init("build")
    result = col.ingest("cats.txt", b"Cats purr.")
    assert col.finalize() == {}

    usage = result.usage
    assert usage.summarizer_input_tokens > 0
    assert usage.summarizer_output_tokens > 0

    cost = usage.cost()
    assert cost.is_complete, cost.unpriced
    assert cost.total_usd > 0
    assert cost.total_usd < Decimal("0.01")  # a ten-byte file is not a cent
    assert next(iter(usage.tokens)).model == LUNA
    col.close()


@live
def test_image_ingestion_prices_against_the_vision_model(store):
    col = store.create("live_image")
    col.init("build")
    result = col.ingest("dot.png", tiny_png())
    col.finalize()

    usage = result.usage
    assert usage.image_input_tokens > 0
    assert usage.summarizer_input_tokens == 0  # vision writes its own description
    cost = usage.cost()
    assert cost.is_complete, cost.unpriced
    assert cost.by("operation")["image"] > 0
    col.close()


@live
@needs_mistral
def test_pdf_ingestion_bills_the_pages_mistral_reports(store):
    col = store.create("live_pdf")
    col.init("build")
    result = col.ingest("report.pdf", tiny_pdf())
    col.finalize()

    usage = result.usage
    assert usage.ocr_pages == 1
    assert usage.ocr["mistral-ocr-latest"].requests == 1

    cost = usage.cost()
    assert cost.is_complete, cost.unpriced
    assert cost.by("operation")["ocr"] == Decimal("0.004")  # $4 / 1000 pages
    col.close()


@live
def test_collection_usage_accumulates_ingestion_and_queries(store):
    col = store.create("live_session")
    col.init("build")
    col.ingest("cats.txt", b"Cats purr.")
    col.finalize()
    after_ingest = col.usage().cost().total_usd

    col.search("a pet that meows")
    assert col.usage().cost().total_usd >= after_ingest  # local embedder is free
    assert col.usage().summarizer_input_tokens > 0
    col.close()

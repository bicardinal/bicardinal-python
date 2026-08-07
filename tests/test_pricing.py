from __future__ import annotations

from decimal import Decimal

import pytest

from bicardinal.extractors.audio import billed_seconds
from bicardinal.office.config import Config
from bicardinal.office.pricing import FLEX
from bicardinal.office.pricing import LONG_CONTEXT_INPUT_THRESHOLD
from bicardinal.office.pricing import STANDARD
from bicardinal.office.pricing import UnknownRate
from bicardinal.office.pricing import embedding_usage
from bicardinal.office.pricing import price
from bicardinal.office.pricing import token_usage
from bicardinal.office.types import TokenKey
from bicardinal.office.types import Usage

LUNA = "gpt-5.6-luna"


class _Details:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _RespUsage:
    def __init__(self, i, o, cached=None, reasoning=None):
        self.input_tokens = i
        self.output_tokens = o
        if cached is not None:
            self.input_tokens_details = _Details(cached_tokens=cached)
        if reasoning is not None:
            self.output_tokens_details = _Details(reasoning_tokens=reasoning)


class _Resp:
    def __init__(self, i, o, cached=None, reasoning=None, tier=None, model=None):
        self.usage = _RespUsage(i, o, cached, reasoning)
        if tier is not None:
            self.service_tier = tier
        if model is not None:
            self.model = model


def _usd(usage: Usage) -> Decimal:
    return usage.cost().total_usd


def _tokens(model, *, i, o, cached=0, tier=STANDARD, long=False, op="summarize"):
    usage = Usage()
    usage.add_tokens(
        TokenKey(operation=op, model=model, service_tier=tier, long_context=long),
        input_tokens=i,
        cached_input_tokens=cached,
        output_tokens=o,
    )
    return usage


# -- defaults --------------------------------------------------------------


def test_luna_is_the_default_model():
    config = Config()
    assert config.summarizer_model == LUNA
    assert config.image_model == LUNA


# -- short vs long context -------------------------------------------------


def test_short_context_rate():
    # 1M in + 100k out at $0.20 / $1.20
    assert _usd(_tokens(LUNA, i=1_000_000, o=100_000)) == Decimal("0.32")


def test_long_context_rate_reprices_the_whole_request():
    # 300k in + 1k out at the long rate: $0.40 in / $1.80 out, applied to every
    # token in the request rather than only the 28k above the threshold.
    cost = _usd(_tokens(LUNA, i=300_000, o=1_000, long=True))
    assert cost == Decimal("0.1218")

    short = _usd(_tokens(LUNA, i=300_000, o=1_000))
    assert cost > short  # and the surcharge is real


def test_long_context_is_exactly_2x_input_and_1_5x_output():
    short = _tokens(LUNA, i=1_000_000, o=0)
    long = _tokens(LUNA, i=1_000_000, o=0, long=True)
    assert _usd(long) == _usd(short) * 2

    short_out = _tokens(LUNA, i=0, o=1_000_000)
    long_out = _tokens(LUNA, i=0, o=1_000_000, long=True)
    assert _usd(long_out) == _usd(short_out) * Decimal("1.5")


def test_threshold_is_per_request_not_per_total():
    """Many small calls must not be billed as one long-context call."""
    small = _Resp(100_000, 500)
    usage = Usage()
    for _ in range(5):  # 500k tokens total, but no single request crosses over
        usage = usage + token_usage(small, operation="summarize", model=LUNA)

    assert usage.summarizer_input_tokens == 500_000
    assert all(not key.long_context for key in usage.tokens)
    assert _usd(usage) == Decimal("0.1030")  # 500k * .20 + 2.5k * 1.20 per 1M

    one_big = token_usage(_Resp(500_000, 2_500), operation="summarize", model=LUNA)
    assert all(key.long_context for key in one_big.tokens)
    assert _usd(one_big) == Decimal("0.2045")  # same tokens, double the input rate


def test_threshold_boundary_is_strictly_above():
    at = token_usage(_Resp(LONG_CONTEXT_INPUT_THRESHOLD, 1), operation="s", model=LUNA)
    over = token_usage(
        _Resp(LONG_CONTEXT_INPUT_THRESHOLD + 1, 1), operation="s", model=LUNA
    )
    assert not next(iter(at.tokens)).long_context
    assert next(iter(over.tokens)).long_context


# -- caching ---------------------------------------------------------------


def test_cached_input_gets_the_discount():
    # 900k of 1M input is a cache read: 100k * $0.20 + 900k * $0.02
    cost = _usd(_tokens(LUNA, i=1_000_000, o=0, cached=900_000))
    assert cost == Decimal("0.038")
    assert cost < _usd(_tokens(LUNA, i=1_000_000, o=0))


def test_cached_tokens_are_a_subset_of_input_not_an_addition():
    usage = token_usage(_Resp(1_000, 0, cached=400), operation="s", model=LUNA)
    tally = next(iter(usage.tokens.values()))
    assert tally.input_tokens == 1_000  # not 1_400
    assert tally.cached_input_tokens == 400


def test_model_without_a_cache_discount_bills_cache_reads_as_input():
    with_cache = _tokens("gpt-5.4-pro", i=1_000_000, o=0, cached=1_000_000)
    without = _tokens("gpt-5.4-pro", i=1_000_000, o=0)
    assert _usd(with_cache) == _usd(without) == Decimal("30")


# -- service tiers ---------------------------------------------------------


def test_flex_is_cheaper_than_standard():
    assert _usd(_tokens(LUNA, i=1_000_000, o=0, tier=FLEX)) == Decimal("0.10")
    assert _usd(_tokens(LUNA, i=1_000_000, o=0)) == Decimal("0.20")


def test_flex_long_context_is_derived_not_free():
    """The flex table prints no long column; it must not fall back to short."""
    flex_long = _usd(_tokens(LUNA, i=1_000_000, o=1_000_000, tier=FLEX, long=True))
    flex_short = _usd(_tokens(LUNA, i=1_000_000, o=1_000_000, tier=FLEX))
    assert flex_long > flex_short
    assert flex_long == Decimal("0.20") + Decimal("0.90")  # 2x in, 1.5x out


def test_tier_reported_by_the_api_wins_over_the_one_requested():
    # A flex request the API served at standard rates must bill as standard.
    usage = token_usage(
        _Resp(100_000, 0, tier="default"),
        operation="summarize",
        model=LUNA,
        service_tier=FLEX,
    )
    assert next(iter(usage.tokens)).service_tier == STANDARD
    assert _usd(usage) == Decimal("0.02")  # standard $0.20/1M, not flex $0.10/1M


def test_mixed_tiers_stay_in_separate_buckets():
    usage = _tokens(LUNA, i=1_000_000, o=0) + _tokens(
        LUNA, i=1_000_000, o=0, tier=FLEX
    )
    assert len(usage.tokens) == 2
    assert _usd(usage) == Decimal("0.30")


# -- model naming ----------------------------------------------------------


def test_dated_snapshot_prices_as_its_base_model():
    usage = token_usage(
        _Resp(100_000, 0, model=f"{LUNA}-2026-04-01"), operation="s", model=LUNA
    )
    cost = usage.cost()
    assert cost.is_complete  # not stranded in `unpriced` by the date suffix
    assert cost.total_usd == Decimal("0.02")


# -- other operations ------------------------------------------------------


def test_audio_is_billed_per_minute():
    usage = Usage()
    usage.add_audio("whisper-1", seconds=90.0, billed_seconds=90)
    assert _usd(usage) == Decimal("0.009")  # 1.5 min * $0.006


def test_audio_rounds_per_request():
    assert billed_seconds(0.4) == 1  # no request is free
    assert billed_seconds(30.4) == 30
    assert billed_seconds(30.6) == 31


def test_ocr_is_billed_per_page():
    usage = Usage()
    usage.add_pages("mistral-ocr-latest", pages=1_000)
    assert _usd(usage) == Decimal("4")


def test_embeddings_are_billed_per_token():
    usage = embedding_usage(model="voyage-4", operation="embed_document", tokens=2_000_000)
    assert _usd(usage) == Decimal("0.12")


def test_local_embedder_usage_is_free_and_absent():
    assert Usage().embeddings == {}
    assert _usd(Usage()) == Decimal(0)


# -- unknown rates ---------------------------------------------------------


def test_unknown_model_is_flagged_rather_than_counted_free():
    cost = _tokens("gpt-does-not-exist", i=1_000_000, o=1_000_000).cost()
    assert cost.total_usd == Decimal(0)
    assert not cost.is_complete
    assert "gpt-does-not-exist" in cost.unpriced[0]


def test_unknown_tier_does_not_silently_bill_as_standard():
    cost = _tokens(LUNA, i=1_000_000, o=0, tier="priority").cost()
    assert not cost.is_complete


def test_strict_raises_on_an_unknown_rate():
    with pytest.raises(UnknownRate):
        price(_tokens("gpt-does-not-exist", i=1, o=1), strict=True)


# -- aggregation -----------------------------------------------------------


def test_usage_add_merges_matching_buckets_and_keeps_distinct_ones():
    a = _tokens(LUNA, i=10, o=1) + _tokens(LUNA, i=20, o=2)
    assert len(a.tokens) == 1
    tally = next(iter(a.tokens.values()))
    assert (tally.requests, tally.input_tokens, tally.output_tokens) == (2, 30, 3)

    b = a + _tokens(LUNA, i=5, o=0, op="image")
    assert len(b.tokens) == 2
    assert b.summarizer_input_tokens == 30
    assert b.image_input_tokens == 5


def test_add_does_not_mutate_either_operand():
    a = _tokens(LUNA, i=10, o=1)
    b = _tokens(LUNA, i=20, o=2)
    a + b
    assert next(iter(a.tokens.values())).input_tokens == 10
    assert next(iter(b.tokens.values())).input_tokens == 20


def test_cost_groups_by_operation():
    usage = _tokens(LUNA, i=1_000_000, o=0) + _tokens(LUNA, i=1_000_000, o=0, op="image")
    by_op = usage.cost().by("operation")
    assert by_op == {"summarize": Decimal("0.20"), "image": Decimal("0.20")}


def test_totals_are_exact_not_floating_point():
    """$0.20/1M is 2e-7 per token; float summation drifts, Decimal does not."""
    usage = Usage()
    for _ in range(1_000):
        usage = usage + _tokens(LUNA, i=1, o=0)
    assert usage.cost().total_usd == Decimal("0.0002")


# -- the rate table itself -------------------------------------------------


def test_published_long_context_rates_match_the_2x_1_5x_rule():
    """Guards the table against a typo when a rate is transcribed by hand."""
    from bicardinal.office.pricing import TOKEN_RATES

    checked = 0
    for (tier, model), rate in TOKEN_RATES.items():
        if rate.long is None:
            continue
        assert rate.long.input == rate.short.input * 2, (tier, model)
        assert rate.long.output == rate.short.output * Decimal("1.5"), (tier, model)
        if rate.short.cached_input is not None:
            assert rate.long.cached_input == rate.short.cached_input * 2, (tier, model)
        checked += 1
    assert checked >= 6  # the models OpenAI publishes both columns for


def test_cached_input_is_never_dearer_than_fresh_input():
    from bicardinal.office.pricing import TOKEN_RATES

    for (tier, model), rate in TOKEN_RATES.items():
        for context in (rate.short, rate.long):
            if context is None or context.cached_input is None:
                continue
            assert context.cached_input < context.input, (tier, model)


def test_flex_undercuts_standard_for_every_model_offering_both():
    from bicardinal.office.pricing import TOKEN_RATES

    for (tier, model), rate in TOKEN_RATES.items():
        if tier != FLEX:
            continue
        standard = TOKEN_RATES.get((STANDARD, model))
        assert standard is not None, model
        assert rate.short.input < standard.short.input, model
        assert rate.short.output < standard.short.output, model


def test_default_models_all_have_a_rate():
    """A shipped default must never land in `unpriced`."""
    config = Config()
    usage = _tokens(config.summarizer_model, i=1, o=1)
    usage = usage + _tokens(config.image_model, i=1, o=1, op="image")
    usage.add_audio(config.transcribe_model, seconds=1.0, billed_seconds=1)
    usage.add_pages(config.ocr_model, pages=1)
    assert usage.cost().is_complete


# -- extractors record what they actually spent ----------------------------


class _FakeImageClient:
    def __init__(self, i=1_500, o=40, cached=0):
        payload = '{"description": "a cat", "transcription": ""}'
        resp = _Resp(i, o, cached=cached)
        resp.output_text = payload
        self.responses = type("R", (), {"create": lambda *a, **kw: resp})()


def test_image_extraction_is_priced_against_the_image_model():
    from bicardinal.extractors.image import ImageExtractor

    result = ImageExtractor(_FakeImageClient(), LUNA).extract(b"\x89PNG\r\n\x1a\n")
    key = next(iter(result.usage.tokens))
    assert (key.operation, key.model) == ("image", LUNA)
    assert result.usage.image_input_tokens == 1_500
    # 1500 * $0.20/1M + 40 * $1.20/1M
    assert result.usage.cost().total_usd == Decimal("0.000348")


def test_image_tokens_do_not_land_in_the_summarizer_bucket():
    from bicardinal.extractors.image import ImageExtractor

    usage = ImageExtractor(_FakeImageClient(), LUNA).extract(b"\x89PNG\r\n\x1a\n").usage
    assert usage.summarizer_input_tokens == 0
    assert usage.image_input_tokens == 1_500


def test_ocr_pages_are_counted_per_request():
    from bicardinal.extractors.pdf import PdfExtractor

    class _Ocr:
        def process(self, **kw):
            pages = [type("P", (), {"markdown": f"page {i}"})() for i in range(3)]
            return type("R", (), {"pages": pages})()

    client = type("M", (), {"ocr": _Ocr()})()
    monkey = PdfExtractor(client, "mistral-ocr-latest")
    monkey._max_pages = 10  # single request; split_pdf needs a real PDF to divide

    import bicardinal.extractors.pdf as pdf_mod

    original = pdf_mod.split_pdf
    pdf_mod.split_pdf = lambda data, **kw: [data]
    try:
        result = monkey.extract(b"%PDF-1.4 fake")
    finally:
        pdf_mod.split_pdf = original

    assert result.usage.ocr_pages == 3
    assert result.usage.ocr["mistral-ocr-latest"].requests == 1
    assert result.usage.cost().total_usd == Decimal("0.012")  # 3 * $4/1000


def test_summarizer_records_model_and_tier_per_call():
    from bicardinal.services.summarizer import Summarizer

    resp = _Resp(100, 20)
    resp.output_text = "a summary"
    client = type(
        "C", (), {"responses": type("R", (), {"create": lambda *a, **kw: resp})()}
    )()

    _, usage, errors = Summarizer(client, LUNA, max_concurrency=2).describe(["a", "b"])
    assert errors == []
    key = next(iter(usage.tokens))
    assert (key.operation, key.model, key.service_tier) == ("summarize", LUNA, STANDARD)
    assert next(iter(usage.tokens.values())).requests == 2
    assert usage.cost().total_usd == Decimal("0.000088")  # 200 in + 40 out


def test_reasoning_tokens_are_recorded_but_not_billed_twice():
    # The Responses API already counts reasoning inside output_tokens.
    usage = token_usage(_Resp(0, 1_000_000, reasoning=400_000), operation="s", model=LUNA)
    assert usage.reasoning_tokens == 400_000
    assert _usd(usage) == Decimal("1.20")  # 1M output, not 1.4M

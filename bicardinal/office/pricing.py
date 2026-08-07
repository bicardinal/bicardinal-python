"""Rate tables and the cost engine.

Every rate here is transcribed from a published price list:

* OpenAI  - https://developers.openai.com/api/docs/pricing (standard/flex/batch)
* Mistral - https://mistral.ai/pricing/api
* Voyage  - https://docs.voyageai.com/docs/pricing

Token rates are USD per 1M tokens, audio is USD per minute, OCR is USD per page.
All arithmetic runs in :class:`~decimal.Decimal`: at $0.20 / 1M tokens a single
token costs 2e-7 USD, and summing millions of those in binary floating point
drifts. Money is never a float here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from decimal import Decimal

from .types import EmbeddingTally
from .types import TokenKey
from .types import Usage

_MILLION = Decimal(1_000_000)
_SECONDS_PER_MINUTE = Decimal(60)

# A request whose *input* exceeds this is billed at the model's long-context
# rate. The surcharge re-prices the whole request -- input and output alike --
# not just the tokens above the line. Because it is a per-request property it
# has to be decided when the call is made; a total summed over many requests
# says nothing about whether any single one of them crossed over.
LONG_CONTEXT_INPUT_THRESHOLD = 272_000

# Wherever OpenAI publishes both tiers the long-context column is exactly 2x
# input and 1.5x output. We use the multipliers only to fill in tables that
# omit a long-context column (flex).
_LONG_INPUT_MULTIPLIER = Decimal("2")
_LONG_OUTPUT_MULTIPLIER = Decimal("1.5")

# "gpt-5.6-luna-2026-04-01" bills at the "gpt-5.6-luna" rate.
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")

STANDARD = "standard"
FLEX = "flex"
BATCH = "batch"

# The API reports the tier that actually served the request; "default"/"auto"
# both bill at standard rates.
_TIER_ALIASES = {"default": STANDARD, "auto": STANDARD, "": STANDARD}


def normalize_tier(tier: str | None) -> str:
    name = (tier or STANDARD).strip().lower()
    return _TIER_ALIASES.get(name, name)


@dataclass(frozen=True)
class TokenRate:
    """USD per 1M tokens for one model, in one service tier, at one context class."""

    input: Decimal
    output: Decimal
    cached_input: Decimal | None = None  # None: no cache discount, bills as input

    def scaled(self) -> "TokenRate":
        """The long-context twin of a short-context rate."""
        return TokenRate(
            input=self.input * _LONG_INPUT_MULTIPLIER,
            output=self.output * _LONG_OUTPUT_MULTIPLIER,
            cached_input=(
                None
                if self.cached_input is None
                else self.cached_input * _LONG_INPUT_MULTIPLIER
            ),
        )

    def cost(self, *, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> Decimal:
        cached = min(cached_input_tokens, input_tokens)  # cached is a subset of input
        fresh = input_tokens - cached
        cached_rate = self.input if self.cached_input is None else self.cached_input
        return (
            Decimal(fresh) * self.input
            + Decimal(cached) * cached_rate
            + Decimal(output_tokens) * self.output
        ) / _MILLION


@dataclass(frozen=True)
class ModelRate:
    """Short- and long-context rates for one model in one service tier."""

    short: TokenRate
    long: TokenRate | None = None  # None: model has no long-context tier

    def at(self, *, long_context: bool) -> TokenRate:
        if not long_context:
            return self.short
        return self.long if self.long is not None else self.short.scaled()


def _rate(inp: str, out: str, cached: str | None = None) -> TokenRate:
    return TokenRate(
        input=Decimal(inp),
        output=Decimal(out),
        cached_input=None if cached is None else Decimal(cached),
    )


def _tiered(short: TokenRate, long: TokenRate) -> ModelRate:
    return ModelRate(short=short, long=long)


# --------------------------------------------------------------------------
# OpenAI text/vision models, keyed (service_tier, model).
#
# Models carrying a long-context tier are written out with both columns as
# published rather than derived, so the table can be diffed against the page.
# --------------------------------------------------------------------------
TOKEN_RATES: dict[tuple[str, str], ModelRate] = {
    # -- standard ----------------------------------------------------------
    (STANDARD, "gpt-5.6-sol"): _tiered(
        _rate("5.00", "30.00", "0.50"), _rate("10.00", "45.00", "1.00")
    ),
    (STANDARD, "gpt-5.6-terra"): _tiered(
        _rate("2.00", "12.00", "0.20"), _rate("4.00", "18.00", "0.40")
    ),
    (STANDARD, "gpt-5.6-luna"): _tiered(
        _rate("0.20", "1.20", "0.02"), _rate("0.40", "1.80", "0.04")
    ),
    (STANDARD, "gpt-5.5"): _tiered(
        _rate("5.00", "30.00", "0.50"), _rate("10.00", "45.00", "1.00")
    ),
    # gpt-5.5-pro is listed "<272K" but the page prints no long row; derived.
    (STANDARD, "gpt-5.5-pro"): ModelRate(_rate("30.00", "180.00")),
    (STANDARD, "gpt-5.4"): _tiered(
        _rate("2.50", "15.00", "0.25"), _rate("5.00", "22.50", "0.50")
    ),
    (STANDARD, "gpt-5.4-mini"): ModelRate(_rate("0.75", "4.50", "0.075")),
    (STANDARD, "gpt-5.4-nano"): ModelRate(_rate("0.20", "1.25", "0.02")),
    (STANDARD, "gpt-5.4-pro"): _tiered(
        _rate("30.00", "180.00"), _rate("60.00", "270.00")
    ),
    (STANDARD, "gpt-5.2"): ModelRate(_rate("1.75", "14.00", "0.175")),
    (STANDARD, "gpt-5.2-pro"): ModelRate(_rate("21.00", "168.00")),
    (STANDARD, "gpt-5.1"): ModelRate(_rate("1.25", "10.00", "0.125")),
    (STANDARD, "gpt-5"): ModelRate(_rate("1.25", "10.00", "0.125")),
    (STANDARD, "gpt-5-mini"): ModelRate(_rate("0.25", "2.00", "0.025")),
    (STANDARD, "gpt-5-nano"): ModelRate(_rate("0.05", "0.40", "0.005")),
    (STANDARD, "gpt-5-pro"): ModelRate(_rate("15.00", "120.00")),
    (STANDARD, "gpt-4.1"): ModelRate(_rate("2.00", "8.00", "0.50")),
    (STANDARD, "gpt-4.1-mini"): ModelRate(_rate("0.40", "1.60", "0.10")),
    (STANDARD, "gpt-4.1-nano"): ModelRate(_rate("0.10", "0.40", "0.025")),
    (STANDARD, "gpt-4o"): ModelRate(_rate("2.50", "10.00", "1.25")),
    (STANDARD, "gpt-4o-mini"): ModelRate(_rate("0.15", "0.60", "0.075")),
    (STANDARD, "o1"): ModelRate(_rate("15.00", "60.00", "7.50")),
    (STANDARD, "o1-pro"): ModelRate(_rate("150.00", "600.00")),
    (STANDARD, "o3"): ModelRate(_rate("2.00", "8.00", "0.50")),
    (STANDARD, "o3-pro"): ModelRate(_rate("20.00", "80.00")),
    (STANDARD, "o3-mini"): ModelRate(_rate("1.10", "4.40", "0.55")),
    (STANDARD, "o4-mini"): ModelRate(_rate("1.10", "4.40", "0.275")),
    # -- flex --------------------------------------------------------------
    # The flex table prints no long-context column; ModelRate derives one from
    # the 2x/1.5x rule so a >272K flex call is never under-counted.
    (FLEX, "gpt-5.6-sol"): ModelRate(_rate("2.50", "15.00", "0.25")),
    (FLEX, "gpt-5.6-terra"): ModelRate(_rate("1.00", "6.00", "0.10")),
    (FLEX, "gpt-5.6-luna"): ModelRate(_rate("0.10", "0.60", "0.01")),
    (FLEX, "gpt-5.4"): ModelRate(_rate("1.25", "7.50", "0.13")),
    (FLEX, "gpt-5.4-mini"): ModelRate(_rate("0.375", "2.25", "0.0375")),
    (FLEX, "gpt-5.4-nano"): ModelRate(_rate("0.10", "0.625", "0.01")),
    (FLEX, "o3"): ModelRate(_rate("1.00", "4.00", "0.25")),
    (FLEX, "o4-mini"): ModelRate(_rate("0.55", "2.20", "0.138")),
    # -- batch -------------------------------------------------------------
    (BATCH, "gpt-5.6-luna"): ModelRate(_rate("0.10", "0.60", "0.01")),
}

# USD per minute of audio.
TRANSCRIPTION_RATES: dict[str, Decimal] = {
    "whisper-1": Decimal("0.006"),
    "gpt-4o-transcribe": Decimal("0.006"),
    "gpt-4o-mini-transcribe": Decimal("0.003"),
    "gpt-transcribe": Decimal("0.0045"),
    "gpt-live-transcribe": Decimal("0.017"),
}

# USD per page. Mistral quotes per 1000 pages.
OCR_RATES: dict[str, Decimal] = {
    "mistral-ocr-latest": Decimal("4") / Decimal(1000),
    "mistral-ocr-4-0": Decimal("4") / Decimal(1000),
}

# USD per 1M tokens. Locally-run sentence-transformers models are not billed
# and never reach this table.
EMBEDDING_RATES: dict[str, Decimal] = {
    "voyage-4-large": Decimal("0.12"),
    "voyage-4": Decimal("0.06"),
    "voyage-4-lite": Decimal("0.02"),
    "voyage-context-4": Decimal("0.12"),
    "voyage-code-3": Decimal("0.18"),
    "voyage-context-3": Decimal("0.18"),
    "voyage-3.5": Decimal("0.06"),
    "voyage-3.5-lite": Decimal("0.02"),
    "text-embedding-3-small": Decimal("0.02"),
    "text-embedding-3-large": Decimal("0.13"),
    "text-embedding-ada-002": Decimal("0.10"),
}


class UnknownRate(LookupError):
    """A billed unit had no published rate to price it with."""


def normalize_model(model: str) -> str:
    """Drop a dated snapshot suffix so snapshots price like their base model."""
    return _SNAPSHOT_SUFFIX.sub("", model.strip())


def _lookup_tokens(model: str, service_tier: str) -> ModelRate | None:
    # Deliberately no fallback across tiers: a tier we hold no table for is
    # reported unpriced rather than quietly billed at standard rates.
    return TOKEN_RATES.get((service_tier, normalize_model(model)))


def _lookup(table: dict[str, Decimal], model: str) -> Decimal | None:
    return table.get(normalize_model(model))


@dataclass(frozen=True)
class CostLine:
    """One priced bucket of usage."""

    kind: str  # tokens | audio | ocr | embedding
    operation: str  # summarize | image | audio | ocr | embed_documents | embed_query
    model: str
    usd: Decimal
    detail: str  # the units this line was billed on
    service_tier: str = ""
    long_context: bool = False


@dataclass
class Cost:
    """An itemized price for a :class:`Usage`."""

    lines: list[CostLine] = field(default_factory=list)
    unpriced: list[str] = field(default_factory=list)

    @property
    def total_usd(self) -> Decimal:
        return sum((line.usd for line in self.lines), Decimal(0))

    @property
    def is_complete(self) -> bool:
        """False when something was used that we had no rate for."""
        return not self.unpriced

    def by(self, attr: str) -> dict[str, Decimal]:
        """Total grouped by any :class:`CostLine` field, e.g. ``"operation"``."""
        out: dict[str, Decimal] = {}
        for line in self.lines:
            k = str(getattr(line, attr))
            out[k] = out.get(k, Decimal(0)) + line.usd
        return out

    def __str__(self) -> str:
        rows = "\n".join(
            f"  {line.operation:<16} {line.model:<22} {line.detail:<34} "
            f"${line.usd:.6f}"
            for line in self.lines
        )
        tail = f"\n  unpriced: {', '.join(self.unpriced)}" if self.unpriced else ""
        return f"${self.total_usd:.6f} total\n{rows}{tail}"


def _price_tokens(key: TokenKey, tally, cost: Cost) -> None:
    rate = _lookup_tokens(key.model, key.service_tier)
    if rate is None:
        cost.unpriced.append(f"{key.model} ({key.service_tier} tokens)")
        return
    usd = rate.at(long_context=key.long_context).cost(
        input_tokens=tally.input_tokens,
        cached_input_tokens=tally.cached_input_tokens,
        output_tokens=tally.output_tokens,
    )
    cached = min(tally.cached_input_tokens, tally.input_tokens)
    cost.lines.append(
        CostLine(
            kind="tokens",
            operation=key.operation,
            model=key.model,
            usd=usd,
            detail=(
                f"{tally.input_tokens - cached} in + {cached} cached "
                f"+ {tally.output_tokens} out"
            ),
            service_tier=key.service_tier,
            long_context=key.long_context,
        )
    )


def price(usage: Usage, *, strict: bool = False) -> Cost:
    """Price a :class:`Usage` into an itemized :class:`Cost`.

    Anything we have no published rate for lands in ``cost.unpriced`` instead of
    being silently counted as free. Pass ``strict=True`` to raise instead.
    """
    cost = Cost()

    for key, tally in usage.tokens.items():
        _price_tokens(key, tally, cost)

    for model, tally in usage.audio.items():
        rate = _lookup(TRANSCRIPTION_RATES, model)
        if rate is None:
            cost.unpriced.append(f"{model} (audio)")
            continue
        cost.lines.append(
            CostLine(
                kind="audio",
                operation="audio",
                model=model,
                usd=Decimal(tally.billed_seconds) * rate / _SECONDS_PER_MINUTE,
                detail=f"{tally.billed_seconds}s over {tally.requests} request(s)",
            )
        )

    for model, tally in usage.ocr.items():
        rate = _lookup(OCR_RATES, model)
        if rate is None:
            cost.unpriced.append(f"{model} (ocr)")
            continue
        cost.lines.append(
            CostLine(
                kind="ocr",
                operation="ocr",
                model=model,
                usd=Decimal(tally.pages) * rate,
                detail=f"{tally.pages} page(s)",
            )
        )

    for (model, operation), tally in usage.embeddings.items():
        rate = _lookup(EMBEDDING_RATES, model)
        if rate is None:
            cost.unpriced.append(f"{model} (embedding)")
            continue
        cost.lines.append(
            CostLine(
                kind="embedding",
                operation=operation,
                model=model,
                usd=Decimal(tally.tokens) * rate / _MILLION,
                detail=f"{tally.tokens} token(s)",
            )
        )

    if strict and cost.unpriced:
        raise UnknownRate("no published rate for: " + ", ".join(cost.unpriced))
    return cost


def token_usage(
    response,
    *,
    operation: str,
    model: str,
    service_tier: str = STANDARD,
) -> Usage:
    """Read one OpenAI Responses result into a :class:`Usage`.

    Picks up the three things a flat token counter loses: how many input tokens
    were cache reads (billed up to 90% cheaper), which service tier actually ran
    the request, and whether this single request crossed the long-context line.
    """
    u = getattr(response, "usage", None)
    input_tokens = _int(getattr(u, "input_tokens", 0))
    output_tokens = _int(getattr(u, "output_tokens", 0))
    cached = _int(getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0))
    reasoning = _int(
        getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0)
    )

    usage = Usage()
    usage.add_tokens(
        TokenKey(
            operation=operation,
            # The response echoes the resolved model, which may be a dated
            # snapshot of the alias we asked for; that is what actually billed.
            model=normalize_model(str(getattr(response, "model", None) or model)),
            service_tier=normalize_tier(
                getattr(response, "service_tier", None) or service_tier
            ),
            long_context=input_tokens > LONG_CONTEXT_INPUT_THRESHOLD,
        ),
        input_tokens=input_tokens,
        cached_input_tokens=min(cached, input_tokens),
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
    )
    return usage


def embedding_usage(*, model: str, operation: str, tokens: int) -> Usage:
    usage = Usage()
    usage.embeddings[(model, operation)] = EmbeddingTally(
        tokens=int(tokens), requests=1
    )
    return usage


def _int(value) -> int:
    return int(value or 0)

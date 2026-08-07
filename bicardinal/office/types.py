from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pricing imports this module; keep the runtime edge one-way
    from .pricing import Cost


class Modality(str, Enum):
    TEXT = "text"
    DOCX = "docx"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class ChunkRecord:
    raw_text: str
    description: str # the text we actually embed
    filename: str
    chunk_index: int


@dataclass
class SearchHit:
    chunk_id: str
    filename: str
    chunk_index: int
    raw_text: str
    description: str
    score: float # brinicle distance; smaller = closer

@dataclass
class Chunk:
    chunk_id: str
    filename: str
    chunk_index: int
    raw_text: str
    description: str

@dataclass
class FileHit:
    filename: str
    score: float # distance of the file's best (min-distance) chunk
    best_chunk: SearchHit


@dataclass(frozen=True)
class TokenKey:
    """What a bucket of tokens costs is decided by all four of these.

    ``long_context`` is per *request*: a request over the long-context
    threshold re-prices in full, so requests either side of the line cannot
    share a bucket. Totalling tokens first and classifying afterwards would
    price a thousand small calls as one huge one.
    """

    operation: str  # summarize | image
    model: str
    service_tier: str  # standard | flex | batch
    long_context: bool = False


@dataclass
class TokenTally:
    requests: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0  # subset of input_tokens, billed cheaper
    cache_write_tokens: int = 0  # subset of input_tokens, may carry a surcharge
    output_tokens: int = 0
    reasoning_tokens: int = 0  # subset of output_tokens, already billed as output

    def __add__(self, other: "TokenTally") -> "TokenTally":
        return TokenTally(
            self.requests + other.requests,
            self.input_tokens + other.input_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.output_tokens + other.output_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class AudioTally:
    requests: int = 0
    seconds: float = 0.0  # true duration
    billed_seconds: int = 0  # per-request durations, each rounded as billed

    def __add__(self, other: "AudioTally") -> "AudioTally":
        return AudioTally(
            self.requests + other.requests,
            self.seconds + other.seconds,
            self.billed_seconds + other.billed_seconds,
        )


@dataclass
class PageTally:
    requests: int = 0
    pages: int = 0

    def __add__(self, other: "PageTally") -> "PageTally":
        return PageTally(self.requests + other.requests, self.pages + other.pages)


@dataclass
class EmbeddingTally:
    requests: int = 0
    tokens: int = 0

    def __add__(self, other: "EmbeddingTally") -> "EmbeddingTally":
        return EmbeddingTally(
            self.requests + other.requests, self.tokens + other.tokens
        )


def _merge(left: dict, right: dict) -> dict:
    out = dict(left)
    for key, tally in right.items():
        out[key] = out[key] + tally if key in out else tally
    return out


@dataclass
class Usage:
    """Billable work, bucketed finely enough to price exactly.

    Tokens are kept per (operation, model, service tier, context class) rather
    than as one running total, because each of those changes the rate.
    """

    tokens: dict[TokenKey, TokenTally] = field(default_factory=dict)
    audio: dict[str, AudioTally] = field(default_factory=dict)  # model -> tally
    ocr: dict[str, PageTally] = field(default_factory=dict)
    embeddings: dict[tuple[str, str], EmbeddingTally] = field(
        default_factory=dict
    )  # (model, operation) -> tally

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            tokens=_merge(self.tokens, other.tokens),
            audio=_merge(self.audio, other.audio),
            ocr=_merge(self.ocr, other.ocr),
            embeddings=_merge(self.embeddings, other.embeddings),
        )

    def add_tokens(self, key: TokenKey, **counts: int) -> None:
        tally = TokenTally(requests=1, **counts)
        self.tokens[key] = self.tokens.get(key, TokenTally()) + tally

    def add_audio(self, model: str, *, seconds: float, billed_seconds: int) -> None:
        tally = AudioTally(requests=1, seconds=seconds, billed_seconds=billed_seconds)
        self.audio[model] = self.audio.get(model, AudioTally()) + tally

    def add_pages(self, model: str, *, pages: int) -> None:
        tally = PageTally(requests=1, pages=pages)
        self.ocr[model] = self.ocr.get(model, PageTally()) + tally

    def cost(self, *, strict: bool = False) -> "Cost":
        """Price this usage. See :func:`bicardinal.office.pricing.price`."""
        from .pricing import price

        return price(self, strict=strict)

    # -- flat views over the buckets, kept for reporting ---------------------
    def _tokens_where(self, operation: str, attr: str) -> int:
        return sum(
            getattr(t, attr) for k, t in self.tokens.items() if k.operation == operation
        )

    @property
    def summarizer_input_tokens(self) -> int:
        return self._tokens_where("summarize", "input_tokens")

    @property
    def summarizer_output_tokens(self) -> int:
        return self._tokens_where("summarize", "output_tokens")

    @property
    def image_input_tokens(self) -> int:
        return self._tokens_where("image", "input_tokens")

    @property
    def image_output_tokens(self) -> int:
        return self._tokens_where("image", "output_tokens")

    @property
    def cached_input_tokens(self) -> int:
        return sum(t.cached_input_tokens for t in self.tokens.values())

    @property
    def cache_write_tokens(self) -> int:
        return sum(t.cache_write_tokens for t in self.tokens.values())

    @property
    def reasoning_tokens(self) -> int:
        return sum(t.reasoning_tokens for t in self.tokens.values())

    @property
    def embedding_tokens(self) -> int:
        return sum(t.tokens for t in self.embeddings.values())

    @property
    def audio_seconds(self) -> float:
        return sum(t.seconds for t in self.audio.values())

    @property
    def ocr_pages(self) -> int:
        return sum(t.pages for t in self.ocr.values())


@dataclass
class AddResult:
    filename: str
    n_chunks: int
    usage: Usage = field(default_factory=Usage)
    errors: list[str] = field(default_factory=list)

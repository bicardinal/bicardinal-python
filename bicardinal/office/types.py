from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum


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


@dataclass
class Usage:
    summarizer_input_tokens: int = 0
    summarizer_output_tokens: int = 0
    image_input_tokens: int = 0
    image_output_tokens: int = 0
    audio_seconds: float = 0.0
    ocr_pages: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.summarizer_input_tokens + other.summarizer_input_tokens,
            self.summarizer_output_tokens + other.summarizer_output_tokens,
            self.image_input_tokens + other.image_input_tokens,
            self.image_output_tokens + other.image_output_tokens,
            self.audio_seconds + other.audio_seconds,
            self.ocr_pages + other.ocr_pages,
        )


@dataclass
class AddResult:
    filename: str
    n_chunks: int
    usage: Usage = field(default_factory=Usage)
    errors: list[str] = field(default_factory=list)

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..extractors.router import Router
from ..services.chunking import chunk_text
from ..services.embedder import Embedder
from ..services.embedder import fuse_documents
from ..services.summarizer import Summarizer
from .config import Config
from .types import ChunkRecord
from .types import Usage

ProgressFn = Callable[[str, int, int], None]  # stage, done, total


def chunk_id_for(filename: str, chunk_index: int, raw_text: str) -> str:
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(b"\x00")
    h.update(str(chunk_index).encode())
    h.update(b"\x00")
    h.update(raw_text.encode())
    return "ch_" + h.hexdigest()


@dataclass
class IngestOutput:
    chunk_ids: list[str]
    vectors: np.ndarray  # (n, dim) float32, L2-normalized
    records: list[ChunkRecord]
    usage: Usage
    errors: list[str]


def build_chunks(
    filename: str,
    data: bytes,
    *,
    router: Router,
    embedder: Embedder,
    summarizer: Summarizer,
    config: Config,
    on_progress: ProgressFn | None = None,
) -> IngestOutput:
    def progress(stage: str, done: int, total: int) -> None:
        if on_progress is not None:
            on_progress(stage, done, total)

    extracted = router.extract(data, filename=filename)  # bytes -> raw text segments
    progress("extract", 1, 1)
    usage = extracted.usage
    errors: list[str] = []

    if (
        extracted.prebuilt_descriptions is not None
    ):  # image: one chunk/segment, vision wrote the description
        raw_texts = extracted.segments
        descriptions = extracted.prebuilt_descriptions
        progress("describe", len(raw_texts), len(raw_texts))
    else:  # text/docx/pdf/audio: window each segment, then summarize
        raw_texts = []
        for seg in extracted.segments:
            raw_texts.extend(
                chunk_text(seg, chunk_size=config.chunk_size, overlap=config.overlap)
            )
        descriptions, desc_usage, desc_errors = summarizer.describe(
            raw_texts, on_tick=lambda d, t: progress("describe", d, t)
        )
        usage = usage + desc_usage
        errors = [f"chunk {i}: {type(e).__name__}: {e}" for i, e in desc_errors]

    index_dim = embedder.dim * 2 if config.dual_encoding else embedder.dim
    if raw_texts:
        desc_vecs = embedder.embed_documents(descriptions)  # embed the DESCRIPTION
        if config.dual_encoding:  # also embed raw text; store both as one vector
            raw_vecs = embedder.embed_documents(raw_texts) # later we can use a different model here.
            vectors = fuse_documents(desc_vecs, raw_vecs)
        else:
            vectors = desc_vecs
    else:
        vectors = np.empty((0, index_dim), dtype=np.float32)
    progress("embed", len(raw_texts), len(raw_texts))

    chunk_ids = [chunk_id_for(filename, i, raw_texts[i]) for i in range(len(raw_texts))]
    records = [
        ChunkRecord(
            raw_text=raw_texts[i],
            description=descriptions[i],
            filename=filename,
            chunk_index=i,
        )
        for i in range(len(raw_texts))
    ]
    return IngestOutput(
        chunk_ids=chunk_ids,
        vectors=vectors,
        records=records,
        usage=usage,
        errors=errors,
    )

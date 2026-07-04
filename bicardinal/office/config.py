from __future__ import annotations

from dataclasses import dataclass

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SUMMARIZER_MODEL = "gpt-5.4-nano"


@dataclass
class Config:
    # chunking
    chunk_size: int = 512
    overlap: float = 0.1

    # ocr
    ocr_model: str = "mistral-ocr-latest"

    # image
    image_model: str = "gpt-5.4"

    # transcribe
    transcribe_model: str = "whisper-1"

    # embedder
    embed_provider: str = "sentence-transformers"  # or "voyage"
    embed_output_dimension: int | None = None
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_batch_size: int = 64
    embed_device: str | None = None
    embed_doc_prompt: str | None = None
    embed_query_prompt: str | None = None

    # summarizer
    summarizer_model: str = DEFAULT_SUMMARIZER_MODEL
    summarizer_max_concurrency: int = 8
    summarizer_reasoning_effort: str = "low"

    # index (HNSW)
    M: int = 16
    ef_construction: int = 200
    efs: int = 64
    build_n_threads: int = 1

    # stores
    shard_count: int = 1

    # search
    default_k: int = 10
    file_scope_threshold: float = 2.0
    n_jobs: int = 1  # parallel shard search; raise it when shard_count > 1

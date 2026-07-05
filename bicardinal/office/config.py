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
    summarizer_use_flex: bool = False # prefer OpenAI "flex"
    summarizer_flex_max_retries: int = 10 # transient-error retries at flex before standard
    summarizer_flex_backoff: float = 1.0 # initial backoff seconds; doubles each retry

    dual_encoding: bool = False  # doubles the index dim
    fusion_weight: float = 0.65  # query-time weight on the description half

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

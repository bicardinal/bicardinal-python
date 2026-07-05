from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return vectors / norms

def fuse_documents(desc_vecs: np.ndarray, raw_vecs: np.ndarray) -> np.ndarray:
    return _l2_normalize(np.concatenate([desc_vecs, raw_vecs], axis=-1))

def fuse_query(q_vec: np.ndarray, weight: float) -> np.ndarray:
    return _l2_normalize(
        np.concatenate([weight * q_vec, (1.0 - weight) * q_vec], axis=-1)
    )

class Embedder(ABC):
    """Common interface for embedding backends."""

    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder(Embedder):
    """sentence-transformers wrapper"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        normalize: bool = True,
        batch_size: int = 64,
        device: str | None = None,
        doc_prompt: str | None = None,
        query_prompt: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self._batch_size = batch_size
        self._doc_prompt = doc_prompt
        self._query_prompt = query_prompt
        self.dim = self._model.get_sentence_embedding_dimension()
        
        if hasattr(self._model, "encode_document"):
            self._encode_doc = self._model.encode_document
            self._doc_prompt = None
        else:
            self._encode_doc = self._model.encode
        if hasattr(self._model, "encode_query"):
            self._encode_query = self._model.encode_query
            self._query_prompt = None
        else:
            self._encode_query = self._model.encode

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode_doc(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            prompt=self._doc_prompt,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode_query(
            text,
            normalize_embeddings=self._normalize,
            prompt=self._query_prompt,
            convert_to_numpy=True,
        ).astype(np.float32)

class VoyageAIEmbedder(Embedder):
    """VoyageAI embedding backend"""

    _MAX_BATCH = 1000

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        output_dimension: int | None = None,
        batch_size: int = 128,
        normalize: bool = True,
        client: object | None = None,
    ) -> None:
        import voyageai

        self._client = client or voyageai.Client(api_key=api_key)
        self._model = model
        self._output_dimension = output_dimension
        self._batch_size = max(1, min(batch_size, self._MAX_BATCH))
        self._normalize = normalize
        self.dim = output_dimension

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = self._client.embed(
                batch,
                model=self._model,
                input_type=input_type,  # "document" indexing / "query" search
                output_dimension=self._output_dimension,
                output_dtype="float",  # only float fits the HNSW index
                truncation=True,  # silently trim over-length inputs
            )
            rows.extend(result.embeddings)
        vectors = np.asarray(rows, dtype=np.float32)
        return _l2_normalize(vectors) if self._normalize else vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "query")[0]


def make_embedder(
    *,
    provider: str = "sentence-transformers",
    model: str,
    api_key: str | None = None,
    batch_size: int = 64,
    normalize: bool = True,
    # sentence-transformers only:
    device: str | None = None,
    doc_prompt: str | None = None,
    query_prompt: str | None = None,
    # voyage only:
    output_dimension: int | None = None,
) -> Embedder:
    """Select an embedding backend by provider name."""
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model,
            normalize=normalize,
            batch_size=batch_size,
            device=device,
            doc_prompt=doc_prompt,
            query_prompt=query_prompt,
        )
    if provider == "voyage":
        return VoyageAIEmbedder(
            model,
            api_key=api_key,
            output_dimension=output_dimension,
            batch_size=batch_size,
            normalize=normalize,
        )
    raise ValueError(
        f"unknown embed_provider {provider!r}; "
        "expected 'sentence-transformers' or 'voyage'"
    )

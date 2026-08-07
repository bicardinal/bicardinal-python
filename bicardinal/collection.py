from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .extractors.router import Router
from .office.config import Config
from .office.exceptions import DocumentNotFound
from .office.exceptions import DuplicateDocument
from .office.pipeline import IngestOutput
from .office.pipeline import ProgressFn
from .office.pipeline import build_chunks
from .office.types import AddResult
from .office.types import FileHit
from .office.types import SearchHit
from .office.types import Chunk
from .office.types import Usage
from .services.embedder import Embedder
from .services.embedder import fuse_query
from .services.summarizer import Summarizer
from .store.catalog import Catalog
from .store.index import Index
from .store.payload import PayloadStore


@dataclass
class _FailedDoc:
    filename: str
    output: IngestOutput
    error: str


@dataclass
class CollectionStatus:
    n_files: int
    n_chunks: int
    filenames: list[str]


class Collection:
    def __init__(
        self,
        path: str | Path,
        *,
        config: Config,
        embedder: Embedder,
        summarizer: Summarizer,
        router: Router,
    ) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._embedder = embedder
        self._summarizer = summarizer
        self._router = router

        self._dual_encoding = config.dual_encoding
        self._fusion_weight = config.fusion_weight
        vector_dim = embedder.dim * 2 if config.dual_encoding else embedder.dim
        self._index = Index(
            self._path / "index",
            vector_dim,
            M=config.M,
            ef_construction=config.ef_construction,
            ef_search=config.efs,
            build_n_threads=config.build_n_threads,
        )

        self._payload = PayloadStore(
            self._path / "payload", shard_count=config.shard_count
        )
        self._catalog = Catalog(self._path / "catalog", shard_count=config.shard_count)
        self._failed: list[_FailedDoc] = []  # write failures, repaired at finalize
        self._usage = Usage()  # everything billed since this handle was opened

    def _write_document(
        self, filename: str, out: IngestOutput, *, upsert: bool = False
    ) -> None:
        for chunk_id, vector, record in zip(out.chunk_ids, out.vectors, out.records):
            self._index.ingest(chunk_id, vector, filename)
        if out.records:
            (self._payload.upsert if upsert else self._payload.insert)(
                dict(zip(out.chunk_ids, out.records))
            )
        self._catalog.add_ids(filename, out.chunk_ids)

    def _project(self, hits: list[tuple[str, float]]) -> list[SearchHit]:
        if not hits:
            return []
        ids = [cid for cid, _ in hits]
        records = self._payload.get(ids)
        out: list[SearchHit] = []
        for (cid, dist), rec in zip(hits, records):
            if rec is None:  # id in index but missing payload — skip defensively
                continue
            out.append(
                SearchHit(
                    chunk_id=cid,
                    filename=rec.filename,
                    chunk_index=rec.chunk_index,
                    raw_text=rec.raw_text,
                    description=rec.description,
                    score=dist,
                )
            )
        return out

    def usage(self) -> Usage:
        """Everything billed through this handle: ingestion and queries both."""
        return self._usage

    def _embed_query(self, query: str, fusion_weight: float | None) -> np.ndarray:
        qvec = self._embedder.embed_query(query)
        self._usage = self._usage + self._embedder.pop_usage()  # queries bill too
        if not self._dual_encoding:
            return qvec
        # if not (0 >= fusion_weight <= 1.0):
        #     raise ValueError("fusion_weight should be within 0-1 range.")
        w = fusion_weight if fusion_weight is not None else self._fusion_weight
        return fuse_query(qvec, w)

    def init(self, mode: str = "insert") -> None:
        self._index.init(mode) # open for staging: build | insert | upsert

    def ingest(
        self,
        filename: str,
        data: bytes,
        *,
        on_progress: ProgressFn | None = None,
    ) -> AddResult:
        if self._catalog.has_file(filename):
            raise DuplicateDocument(filename)
        out = build_chunks(
            filename,
            data,
            router=self._router,
            embedder=self._embedder,
            summarizer=self._summarizer,
            config=self._config,
            on_progress=on_progress,
        )
        errors = list(out.errors)
        self._usage = self._usage + out.usage
        try:
            self._write_document(filename, out)
            if on_progress is not None:
                on_progress("write", len(out.records), len(out.records))
        except Exception as e:  # defer; finalize purges partial ids and replays
            self._failed.append(_FailedDoc(filename, out, str(e)))
            errors.append(f"write deferred to finalize: {e}")
        return AddResult(filename, len(out.records), out.usage, errors)

    def finalize(self, *, max_retries: int = 3) -> dict[str, str]:
        self._index.finalize()  # build everything staged in the normal pass
        retry = self._failed
        self._failed = []
        attempt = 0
        while retry and attempt < max_retries:
            attempt += 1
            self._index.init("upsert")  # replay overwrites partial entries by id
            still: list[_FailedDoc] = []
            for fd in retry:
                try:
                    self._catalog.delete_file(fd.filename)  # idempotent catalog replay
                    self._write_document(fd.filename, fd.output, upsert=True)
                except Exception as e:
                    fd.error = str(e)
                    still.append(fd)
            self._index.finalize()  # build the repair delta
            retry = still
        return {
            fd.filename: fd.error for fd in retry
        }  # terminal failures; {} = all repaired

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        n_jobs: int | None = None,
        efs: int | None = None,
        fusion_weight: float | None = None,
    ) -> list[SearchHit]:
        if not self._catalog.get_files():
            return []
        k = k if k is not None else self._config.default_k
        n_jobs = n_jobs if n_jobs is not None else self._config.n_jobs
        efs = efs if efs is not None else self._config.efs
        qvec = self._embed_query(query, fusion_weight)
        hits = self._index.search_with_distance(qvec, k, n_jobs=n_jobs, efs=efs)
        return self._project(hits)

    def search_in_file(
        self,
        query: str,
        filename: str,
        k: int | None = None,
        *,
        n_jobs: int | None = None,
        efs: int | None = None,
        exact: bool = True,
        fusion_weight: float | None = None,
    ) -> list[SearchHit]:
        if not self._catalog.has_file(filename):
            raise DocumentNotFound(filename)
        k = k if k is not None else self._config.default_k
        n_jobs = n_jobs if n_jobs is not None else self._config.n_jobs
        efs = efs if efs is not None else self._config.efs
        qvec = self._embed_query(query, fusion_weight)
        hits = self._index.search_with_distance(
            qvec,
            k,
            category=filename,
            threshold=self._config.file_scope_threshold,
            n_jobs=n_jobs,
            efs=efs,
        )
        results = self._project(hits)
        if exact:
            results = [
                h for h in results if h.filename == filename
            ] # drop category hash-collisions
        return results

    def most_similar_files(
        self,
        query: str,
        k: int | None = None,
        *,
        candidate_k: int = 100,
        n_jobs: int | None = None,
        efs: int | None = None,
        fusion_weight: float | None = None,
    ) -> list[FileHit]:
        if not self._catalog.get_files():
            return []
        k = k if k is not None else self._config.default_k
        n_jobs = n_jobs if n_jobs is not None else self._config.n_jobs
        efs = efs if efs is not None else self._config.efs
        qvec = self._embed_query(query, fusion_weight)
        hits = self._index.search_with_distance(
            qvec, candidate_k, n_jobs=n_jobs, efs=efs
        )
        best: dict[str, SearchHit] = {}
        for h in self._project(hits):
            if (
                h.filename not in best
            ):  # ascending distance -> first per file is its best
                best[h.filename] = h
        ranked = sorted(
            best.values(), key=lambda h: h.score
        )  # files by their best chunk
        return [
            FileHit(filename=h.filename, score=h.score, best_chunk=h)
            for h in ranked[:k]
        ]

    def status(self) -> CollectionStatus:
        files = self._catalog.get_files()
        n_chunks = sum(len(self._catalog.get_ids(f)) for f in files)
        return CollectionStatus(n_files=len(files), n_chunks=n_chunks, filenames=files)

    def delete(self, filename: str) -> None:
        if not self._catalog.has_file(filename):
            raise DocumentNotFound(filename)
        ids = self._catalog.get_ids(filename)
        self._index.delete(ids)
        self._payload.delete(ids)
        self._catalog.delete_file(
            filename
        )  # catalog last: file stays consistent until purge completes

    def read_chunks(
        self,
        filename: str,
        *,
        start: int = 0,
        limit: int | None = None,
    ) -> list[Chunk]:
        if not self._catalog.has_file(filename):
            raise DocumentNotFound(filename)
        ids = self._catalog.get_ids(filename)
        records = self._payload.get(ids)
        chunks = [
            Chunk(
                chunk_id=cid,
                filename=rec.filename,
                chunk_index=rec.chunk_index,
                raw_text=rec.raw_text,
                description=rec.description,
            )
            for cid, rec in zip(ids, records)
            if rec is not None
        ]
        chunks.sort(key=lambda c: c.chunk_index)
        stop = None if limit is None else start + limit
        return chunks[start:stop]

    def close(self) -> None:
        self._index.close()
        self._payload.close()
        self._catalog.close()

    def destroy(self) -> None:
        self._index.destroy()
        self._payload.destroy()
        self._catalog.destroy()

    def __enter__(self) -> "Collection":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

from __future__ import annotations

import shutil
from pathlib import Path

from mistralai.client import Mistral
from openai import OpenAI

from .collection import Collection
from .collection import CollectionStatus
from .extractors.audio import AudioExtractor
from .extractors.docx import DocxExtractor
from .extractors.image import ImageExtractor
from .extractors.pdf import PdfExtractor
from .extractors.router import Router
from .extractors.text import TextExtractor
from .office.config import DEFAULT_EMBED_MODEL
from .office.config import Config
from .office.exceptions import BicardinalError
from .office.exceptions import CollectionExists
from .office.exceptions import CollectionNotFound
from .office.exceptions import DocumentNotFound
from .office.exceptions import DuplicateDocument
from .office.exceptions import EmptyFile
from .office.exceptions import ExtractionError
from .office.exceptions import UnsupportedFileType
from .office.types import AddResult
from .office.types import FileHit
from .office.types import Modality
from .office.types import SearchHit
from .office.types import Chunk
from .office.types import Usage
from .services.embedder import make_embedder
from .services.summarizer import Summarizer


class Bicardinal:
    """Owns a directory of collections and the shared models/clients they use."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        config: Config | None = None,
        openai_api_key: str | None = None,
        mistral_api_key: str | None = None,
        voyage_api_key: str | None = None,
    ) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._config = config or Config()

        openai_client = OpenAI(
            api_key=openai_api_key
        )  # shared across summarizer/image/audio
        mistral_client = Mistral(api_key=mistral_api_key)

        if (
            self._config.embed_provider == "voyage"
            and self._config.embed_model == DEFAULT_EMBED_MODEL
        ):
            raise ValueError(
                "embed_provider='voyage' requires embed_model to be a Voyage model "
                "(e.g. 'voyage-4'); it is still the sentence-transformers default."
            )
        self._embedder = make_embedder(
            provider=self._config.embed_provider,
            model=self._config.embed_model,
            api_key=voyage_api_key,
            batch_size=self._config.embed_batch_size,
            output_dimension=self._config.embed_output_dimension,
            device=self._config.embed_device,
            doc_prompt=self._config.embed_doc_prompt,
            query_prompt=self._config.embed_query_prompt,
        )

        self._summarizer = Summarizer(
            openai_client,
            self._config.summarizer_model,
            max_concurrency=self._config.summarizer_max_concurrency,
            reasoning_effort=self._config.summarizer_reasoning_effort,
        )
        self._router = Router(
            {
                Modality.TEXT: TextExtractor(),
                Modality.DOCX: DocxExtractor(),
                Modality.PDF: PdfExtractor(mistral_client, self._config.ocr_model),
                Modality.IMAGE: ImageExtractor(openai_client, self._config.image_model),
                Modality.AUDIO: AudioExtractor(
                    openai_client, self._config.transcribe_model
                ),
            }
        )

    def _collection_path(self, name: str) -> Path:
        if not name.lower().replace("_", "").isalnum():
            raise ValueError(
                f"invalid collection name. Only [A-z0-9_] are supported: {name!r}"
            )
        return self._root / name

    def _build_collection(self, path: Path) -> Collection:
        return Collection(
            path,
            config=self._config,
            embedder=self._embedder,
            summarizer=self._summarizer,
            router=self._router,
        )

    def _is_collection(self, path: Path) -> bool:
        return path.is_dir() and any(path.iterdir())  # exists, and has store files

    def create(self, name: str) -> Collection:
        path = self._collection_path(name)
        if path.exists():
            raise CollectionExists(name)
        return self._build_collection(
            path
        )  # Collection.__init__ mkdirs + inits empty stores

    def open(self, name: str) -> Collection:
        path = self._collection_path(name)
        if not self._is_collection(path):
            raise CollectionNotFound(name)
        return self._build_collection(path)

    def list(self) -> list[str]:
        return sorted(p.name for p in self._root.iterdir() if self._is_collection(p))

    def delete(self, name: str) -> None:
        path = self._collection_path(name)
        if not self._is_collection(path):
            raise CollectionNotFound(name)
        col = self._build_collection(path)
        col.destroy()  # clear brinicle stores
        shutil.rmtree(path, ignore_errors=True)  # remove the now-empty directory

    def destroy(self) -> None:
        for name in self.list():  # snapshot first, then mutate the filesystem
            self.delete(name)


__all__ = [
    "Bicardinal",
    "Collection",
    "Config",
    "CollectionStatus",
    "AddResult",
    "SearchHit",
    "FileHit",
    "Chunk",
    "Usage",
    "Modality",
    "BicardinalError",
    "DuplicateDocument",
    "DocumentNotFound",
    "CollectionExists",
    "CollectionNotFound",
    "UnsupportedFileType",
    "EmptyFile",
    "ExtractionError",
]

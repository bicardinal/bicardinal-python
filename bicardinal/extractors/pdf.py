from __future__ import annotations

import base64
from io import BytesIO

from mistralai.client import Mistral
from pypdf import PdfReader
from pypdf import PdfWriter

from ..office.config import DEFAULT_OCR_MODEL
from ..office.exceptions import ExtractionError
from ..office.types import Modality
from ..office.types import Usage
from .base import Extractor
from .base import ExtractResult


def _write_pages(pages) -> bytes:
    writer = PdfWriter()
    for p in pages:
        writer.add_page(p)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def split_pdf(data: bytes, *, max_pages: int, max_bytes: int) -> list[bytes]:
    reader = PdfReader(BytesIO(data))
    pages = list(reader.pages)
    if len(pages) <= max_pages and len(data) <= max_bytes:
        return [data]

    out: list[bytes] = []

    def emit(group) -> None:
        blob = _write_pages(group)
        if (
            len(blob) <= max_bytes or len(group) == 1
        ):  # fits, or unsplittable single page
            out.append(blob)
        else:
            mid = len(group) // 2  # too big -> halve and recurse
            emit(group[:mid])
            emit(group[mid:])

    for i in range(0, len(pages), max_pages):  # page-count cut first
        emit(pages[i : i + max_pages])
    return out


class PdfExtractor(Extractor):
    modality = Modality.PDF

    def __init__(
        self,
        client: Mistral,
        model: str = DEFAULT_OCR_MODEL,
        *,
        max_pages: int = 1000,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._max_pages = max_pages
        self._max_bytes = max_bytes

    def extract(self, data: bytes, *, filename: str | None = None) -> ExtractResult:
        segments: list[str] = []
        usage = Usage()
        try:
            for chunk in split_pdf(
                data, max_pages=self._max_pages, max_bytes=self._max_bytes
            ):
                b64 = base64.b64encode(chunk).decode()
                resp = self._client.ocr.process(
                    model=self._model,
                    document={
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{b64}",
                    },
                )
                pages = list(resp.pages)
                # Mistral bills per page, per request; a split PDF is several
                # requests whose page counts add up to the whole document.
                usage.add_pages(self._model, pages=len(pages))
                segments.extend(page.markdown for page in pages)
        except Exception as e:
            raise ExtractionError(f"PDF OCR failed: {e}") from e
        return ExtractResult(segments=segments, modality=self.modality, usage=usage)

from __future__ import annotations

from io import BytesIO

from openai import OpenAI

from ..office.config import DEFAULT_TRANSCRIBE_MODEL
from ..office.exceptions import ExtractionError
from ..office.types import Modality
from ..office.types import Usage
from .base import Extractor
from .base import ExtractResult


def billed_seconds(seconds: float) -> int:
    """Transcription bills to the nearest second, and no request is free."""
    return max(1, round(seconds))


def split_audio(
    data: bytes, *, filename: str | None, chunk_seconds: int, max_bytes: int
) -> tuple[list[tuple[bytes, str, float]], float]:
    """Split into transcribable pieces, each tagged with its own duration.

    Transcription is billed per request, so the caller needs the per-piece
    durations and not just the total.
    """
    from pydub import AudioSegment  # optional [audio] dep; needs ffmpeg

    audio = AudioSegment.from_file(BytesIO(data))
    seconds = len(audio) / 1000.0
    if len(data) <= max_bytes:  # fits; keep original bytes
        return [(data, filename or "audio.mp3", seconds)], seconds

    out: list[tuple[bytes, str, float]] = []

    def emit(segment, idx: int) -> None:
        buf = BytesIO()
        segment.export(buf, format="mp3")
        blob = buf.getvalue()
        if (
            len(blob) <= max_bytes or len(segment) <= 1000
        ):  # fits, or <=1s, unsplittable
            out.append((blob, f"chunk_{idx}.mp3", len(segment) / 1000.0))
        else:
            half = len(segment) // 2  # too big -> halve and recurse
            emit(segment[:half], idx)
            emit(segment[half:], idx)

    window = chunk_seconds * 1000  # ms
    for i in range(0, len(audio), window):
        emit(audio[i : i + window], i // window)
    return out, seconds


class AudioExtractor(Extractor):
    modality = Modality.AUDIO

    def __init__(
        self,
        client: OpenAI,
        model: str = DEFAULT_TRANSCRIBE_MODEL,
        *,
        chunk_seconds: int = 600,
        max_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._chunk_seconds = chunk_seconds
        self._max_bytes = max_bytes

    def extract(self, data: bytes, *, filename: str | None = None) -> ExtractResult:
        usage = Usage()
        try:
            chunks, _ = split_audio(
                data,
                filename=filename,
                chunk_seconds=self._chunk_seconds,
                max_bytes=self._max_bytes,
            )
            texts: list[str] = []
            for blob, name, seconds in chunks:
                resp = self._client.audio.transcriptions.create(
                    model=self._model, file=(name, blob)
                )
                # Round per request, not once over the total: each request is
                # rounded on its own before it is billed.
                usage.add_audio(
                    self._model,
                    seconds=seconds,
                    billed_seconds=billed_seconds(seconds),
                )
                texts.append(resp.text)
        except Exception as e:
            raise ExtractionError(f"audio transcription failed: {e}") from e
        transcript = " ".join(t.strip() for t in texts if t.strip())
        return ExtractResult(segments=[transcript], modality=self.modality, usage=usage)

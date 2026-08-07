from __future__ import annotations

import base64
import json

from openai import OpenAI

from ..office.config import DEFAULT_IMAGE_MODEL
from ..office.exceptions import ExtractionError
from ..office.pricing import token_usage
from ..office.types import Modality
from .base import Extractor
from .base import ExtractResult

IMAGE_PROMPT = (
    "Describe the image in 1-3 sentences for semantic search, subject, content, "
    "notable detail. Also transcribe verbatim any text in the image; if there is "
    "none, use an empty string."
)

IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "transcription": {"type": "string"},
    },
    "required": ["description", "transcription"],
    "additionalProperties": False,
}


def _image_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"  # fallback; router already restricts to known image types


class ImageExtractor(Extractor):
    modality = Modality.IMAGE

    def __init__(
        self, client: OpenAI, model: str = DEFAULT_IMAGE_MODEL, *, detail: str = "auto"
    ) -> None:
        self._client = client
        self._model = model
        self._detail = detail

    def extract(self, data: bytes, *, filename: str | None = None) -> ExtractResult:
        b64 = base64.b64encode(data).decode()
        try:
            resp = self._client.responses.create(
                model=self._model,
                instructions=IMAGE_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:{_image_mime(data)};base64,{b64}",
                                "detail": self._detail,
                            }
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "image_extract",
                        "schema": IMAGE_SCHEMA,
                        "strict": True,
                    }
                },
            )
            parsed = json.loads(resp.output_text)
            description = parsed["description"].strip()
            transcription = parsed["transcription"].strip()
        except Exception as e:
            raise ExtractionError(f"image extraction failed: {e}") from e
        # Vision input arrives as image tokens already counted in input_tokens,
        # so the model's normal text rate prices the call.
        usage = token_usage(resp, operation="image", model=self._model)
        return ExtractResult(
            segments=[transcription],
            modality=self.modality,
            prebuilt_descriptions=[description],
            usage=usage,
        )

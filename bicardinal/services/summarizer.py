from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from openai import OpenAI

from ..office.types import Usage

DESCRIBE_PROMPT = (
    "Summarize the text below in 1-3 sentences for semantic search. "
    "Capture its main topics, entities, and specifics. "
    "Output only the summary."
    """We use your output for RAG systems.
Focus on:
- Key topics and concepts
- Main arguments or findings
- Important details that distinguish this section

Provide a concise summary that can be a good representation of the whole content.
If the given content is too small to be summarized (< 120 tokens), just rephrase the content."""
)


class Summarizer:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-5.4-nano",
        *,
        max_concurrency: int = 8,
        reasoning_effort: str = "low",
    ) -> None:
        self._client = client
        self._model = model
        self._max_concurrency = max_concurrency
        self._reasoning_effort = reasoning_effort

    def _describe_one(self, text: str) -> tuple[str, Usage]:
        resp = self._client.responses.create(
            model=self._model,
            reasoning={"effort": self._reasoning_effort},
            instructions=DESCRIBE_PROMPT,
            input=f"CONTEXT: {text}",
        )
        description = resp.output_text.strip()
        usage = Usage(
            summarizer_input_tokens=resp.usage.input_tokens,
            summarizer_output_tokens=resp.usage.output_tokens,
        )
        return description, usage

    def describe(
        self,
        chunks: list[str],
        *,
        on_tick: Callable[[int, int], None] | None = None,
    ) -> tuple[list[str], Usage, list[tuple[int, Exception]]]:
        n = len(chunks)
        descriptions: list[str] = [""] * n
        usage = Usage()
        errors: list[tuple[int, Exception]] = []
        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            futures = {
                pool.submit(self._describe_one, t): i for i, t in enumerate(chunks)
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    d, u = fut.result()
                    descriptions[i] = d
                    usage = usage + u
                except Exception as e:
                    descriptions[i] = chunks[i] # fall back to raw text
                    errors.append((i, e))
                done += 1
                if on_tick is not None:
                    on_tick(done, n)
        errors.sort() # by index; completion order is arbitrary
        return descriptions, usage, errors

from __future__ import annotations

import random
import time

from openai import APIConnectionError
from openai import APITimeoutError
from openai import InternalServerError
from openai import OpenAI
from openai import RateLimitError

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from openai import OpenAI

from ..office.types import Usage

_FLEX_RETRYABLE = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

# OpenAI recommends a 15-minute request timeout for flex (SDK default is 10).
_FLEX_TIMEOUT_SECONDS = 900.0


DESCRIBE_PROMPT = (
    "Summarize the text below in 1-3 sentences for semantic search."
    "Capture its main topics, entities, and specifics. "
    "Output only the summary."
    """We use your output for RAG systems.
Focus on:
- Key topics and concepts
- Main arguments or findings
- Important details that distinguish this section

Provide a concise summary that can be a good representation of the whole content.
If the given content is too small to be summarized (< 120 tokens), just rephrase the content.
One key point is that you should provide the summary in English regardless of the content language."""
)


class Summarizer:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-5.4-nano",
        *,
        max_concurrency: int = 8,
        reasoning_effort: str = "low",
        use_flex: bool = False,
        flex_max_retries: int = 3,
        flex_backoff: float = 0.5,
    ) -> None:
        self._client = client
        self._model = model
        self._max_concurrency = max_concurrency
        self._reasoning_effort = reasoning_effort
        self._use_flex = use_flex
        self._flex_max_retries = flex_max_retries
        self._flex_backoff = flex_backoff


    def _call(
        self, text: str, *, service_tier: str | None, timeout: float | None = None
    ) -> tuple[str, Usage]:
        kwargs: dict = {}
        if service_tier is not None:
            kwargs["service_tier"] = service_tier
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = self._client.responses.create(
            model=self._model,
            reasoning={"effort": self._reasoning_effort},
            instructions=DESCRIBE_PROMPT,
            input=f"CONTEXT: {text}",
            **kwargs,
        )
        description = resp.output_text.strip()
        usage = Usage(
            summarizer_input_tokens=resp.usage.input_tokens,
            summarizer_output_tokens=resp.usage.output_tokens,
        )
        return description, usage

    def _describe_flex(self, text: str) -> tuple[str, Usage]:
        delay = self._flex_backoff
        last_exc: Exception | None = None
        for attempt in range(self._flex_max_retries + 1):
            try:
                return self._call(
                    text, service_tier="flex", timeout=_FLEX_TIMEOUT_SECONDS
                )
            except _FLEX_RETRYABLE as e:
                last_exc = e
                if attempt == self._flex_max_retries:
                    break
                time.sleep(delay * (0.5 + random.random())) # backoff + jitter
                delay *= 2
        assert last_exc is not None
        raise last_exc

    def _describe_one(self, text: str) -> tuple[str, Usage]:
        if self._use_flex:
            try:
                return self._describe_flex(text)
            except Exception:
                pass
        return self._call(text, service_tier=None)

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

"""
benchmark/request_runner.py

Async client for a vLLM OpenAI-compatible endpoint. Each request returns a
RequestResult with TTFT, TPOT, end-to-end latency, output token count, and a
classified failure_reason (ok / timeout / oom / http_error / other) so the
sweep driver can apply the 3-tier feasibility rule.

Why streaming:
  TTFT is meaningful only with stream=True. For non-streaming requests, the
  client only sees the response after decode finishes, so TTFT == latency.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

# vLLM error strings vary across versions; this regex is intentionally broad.
# Used to reclassify a generic 5xx as an OOM so feasibility labels stay honest.
_OOM_PATTERNS = re.compile(
    r"(out of memory|cuda.*oom|kv cache.*full|cannot allocate|no available memory|"
    r"engine.*dead|cuda error)",
    re.IGNORECASE,
)


# ------------------------------------------------------------------------------
# Public types
# ------------------------------------------------------------------------------

@dataclass
class RunnerConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"                 # vLLM ignores key but SDK requires one
    model: str = "qwen2.5-14b-awq"         # must match --served-model-name on server
    max_output_tokens: int = 128           # fixed across the sweep so TPOT is comparable
    temperature: float = 0.0               # we measure timing, not quality
    request_timeout_s: float = 120.0


@dataclass
class RequestResult:
    ok: bool
    ttft_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    output_tokens: Optional[int] = None
    failure_reason: str = "ok"             # ok | timeout | oom | http_error | other
    error_msg: str = ""


# ------------------------------------------------------------------------------
# Client construction
# ------------------------------------------------------------------------------

def make_client(cfg: RunnerConfig) -> AsyncOpenAI:
    """
    Build an AsyncOpenAI client with explicit httpx timeouts and zero retries.
    We disable transport-level retries because we want failures to surface as
    measured outcomes (timeout / http_error), not be silently masked.
    """
    http_client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(retries=0),
        timeout=httpx.Timeout(cfg.request_timeout_s, connect=10.0),
    )
    return AsyncOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        http_client=http_client,
        max_retries=0,
    )


# ------------------------------------------------------------------------------
# Single-request send + measure
# ------------------------------------------------------------------------------

async def send_request(
    client: AsyncOpenAI,
    cfg: RunnerConfig,
    prompt: str,
) -> RequestResult:
    """
    Fire one streaming chat-completion and time it.

    Timing model:
        t0          = just before .create() call
        t_first     = wall time of the FIRST chunk that carries non-empty content
                      (role-only deltas are skipped -- see comment below)
        t_end       = wall time after the stream is fully drained

        ttft_ms     = (t_first - t0) * 1000
        latency_ms  = (t_end   - t0) * 1000
        tpot_ms     = (t_end - t_first) * 1000 / max(out_tokens - 1, 1)
                      (excludes the first token, which is part of TTFT)

    Output tokens come from the server's usage block (stream_options=include_usage).
    Falling back to chunk-counting is incorrect under load because vLLM packs
    multiple tokens per SSE frame when the scheduler is saturated.
    """
    t0 = time.perf_counter()
    t_first: Optional[float] = None
    out_tokens_from_usage: Optional[int] = None
    content_chunks = 0

    try:
        stream = await client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
            stream=True,
            stream_options={"include_usage": True},
            # vLLM extension: forces decode to run for exactly max_tokens steps
            # regardless of EOS, so every cell in the sweep does the same amount
            # of decode work and TPOT comparisons are fair.
            extra_body={"ignore_eos": True},
            timeout=cfg.request_timeout_s,
        )

        async for chunk in stream:
            # The final chunk (when include_usage=True) has empty choices and
            # carries the authoritative completion_tokens count.
            if getattr(chunk, "usage", None) is not None:
                out_tokens_from_usage = chunk.usage.completion_tokens
                continue

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            # Skip role-only / empty deltas. The OpenAI streaming protocol
            # opens with a frame that sets role="assistant" but has content="".
            # Stamping TTFT here would underestimate prefill time.
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                content_chunks += 1

    except (asyncio.TimeoutError, APITimeoutError) as exc:
        return RequestResult(ok=False, failure_reason="timeout", error_msg=str(exc))
    except APIStatusError as exc:
        msg = f"{exc.status_code}: {getattr(exc, 'message', str(exc))}"
        reason = "oom" if _OOM_PATTERNS.search(msg) else "http_error"
        return RequestResult(ok=False, failure_reason=reason, error_msg=msg)
    except APIConnectionError as exc:
        # Connection refused / reset typically means the engine died.
        # We don't auto-promote to "oom" without evidence; the sweep driver
        # can correlate with gpu_monitor peaks if it wants to upgrade the label.
        return RequestResult(ok=False, failure_reason="http_error", error_msg=str(exc))
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        reason = "oom" if _OOM_PATTERNS.search(msg) else "other"
        return RequestResult(ok=False, failure_reason=reason, error_msg=msg)

    t_end = time.perf_counter()

    if t_first is None:
        # 200 OK but no content emitted -- treat as failure so feasibility is honest.
        return RequestResult(
            ok=False, failure_reason="other", error_msg="empty response stream"
        )

    latency_ms = (t_end - t0) * 1000.0
    ttft_ms = (t_first - t0) * 1000.0
    out_tokens = out_tokens_from_usage if out_tokens_from_usage is not None else content_chunks
    # TPOT is undefined for single-token outputs.
    tpot_ms = (
        (t_end - t_first) * 1000.0 / (out_tokens - 1)
        if out_tokens and out_tokens > 1
        else None
    )

    return RequestResult(
        ok=True,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        tpot_ms=tpot_ms,
        output_tokens=out_tokens,
        failure_reason="ok",
    )


# ------------------------------------------------------------------------------
# Concurrent burst -- the unit the sweep driver actually calls
# ------------------------------------------------------------------------------

async def send_concurrent(
    client: AsyncOpenAI,
    cfg: RunnerConfig,
    prompts: list[str],
) -> list[RequestResult]:
    """
    Launch len(prompts) requests in parallel and gather their results.

    asyncio.gather schedules all coroutines on the event loop within a few
    microseconds of each other, so the server sees a concurrency burst of
    exactly len(prompts) for the duration of the slowest request. This is
    what the spec means by "concurrency = N".

    return_exceptions=False is intentional: send_request never raises -- it
    converts everything into RequestResult(ok=False, ...). If gather ever
    raises, that's a bug in this module, not a server failure.
    """
    return await asyncio.gather(*(send_request(client, cfg, p) for p in prompts))


# ------------------------------------------------------------------------------
# Smoke test (manual): python -m benchmark.request_runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    async def _main() -> None:
        cfg = RunnerConfig()
        client = make_client(cfg)
        try:
            results = await send_concurrent(client, cfg, ["Say hello in one word."] * 2)
            for i, r in enumerate(results):
                print(f"[{i}] {r}")
        finally:
            await client.close()

    asyncio.run(_main())

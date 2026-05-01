"""
tools/mock_vllm_server.py

Tiny aiohttp app that pretends to be a vLLM OpenAI-compatible server.
Use for local development of the benchmark client without a real GPU.

What it does:
    GET  /v1/models             -> {"data": [{"id": SERVED_NAME, ...}]}
    POST /v1/chat/completions   -> streaming SSE chat-completion response

What it does NOT do:
    - Run any model. Output content is fixed filler text.
    - Tokenize the input. completion_tokens in the usage block == max_tokens
      (the client uses this for TPOT math, which is the only thing we need
      to be self-consistent for plumbing tests).
    - Reflect real GPU behavior. Latencies are configured, not measured.

Why aiohttp and not FastAPI:
    aiohttp is already pinned in requirements.txt for the benchmark client.
    Reusing it keeps the dev environment lean -- no extra dependency just
    to mock a server.

Config (env vars, all optional):
    HOST                 default 0.0.0.0
    PORT                 default 8000
    SERVED_NAME          default qwen2.5-14b-awq  (advertised by /v1/models)
    MOCK_TTFT_MS         default 50               simulated prefill delay
    MOCK_TPOT_MS         default 10               simulated per-token decode delay
    MOCK_FAIL_MODE       default none             one of: none|oom|http_500|timeout|empty
    MOCK_FAIL_AFTER_REQ  default 0                fail-mode kicks in after N successful requests
                                                  (0 = fail from the very first request)

Run:
    python -m tools.mock_vllm_server

Quick smoke:
    curl -s http://localhost:8000/v1/models | python -m json.tool
    python -m benchmark.request_runner       # against a default-port mock
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from aiohttp import web

# ------------------------------------------------------------------------------
# Config (read once at import; restart the server to change)
# ------------------------------------------------------------------------------

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
SERVED_NAME = os.getenv("SERVED_NAME", "qwen2.5-14b-awq")
MOCK_TTFT_MS = float(os.getenv("MOCK_TTFT_MS", "50"))
MOCK_TPOT_MS = float(os.getenv("MOCK_TPOT_MS", "10"))
MOCK_FAIL_MODE = os.getenv("MOCK_FAIL_MODE", "none").lower()
MOCK_FAIL_AFTER_REQ = int(os.getenv("MOCK_FAIL_AFTER_REQ", "0"))

_VALID_FAIL_MODES = {"none", "oom", "http_500", "timeout", "empty"}
if MOCK_FAIL_MODE not in _VALID_FAIL_MODES:
    raise SystemExit(
        f"MOCK_FAIL_MODE={MOCK_FAIL_MODE!r} not in {sorted(_VALID_FAIL_MODES)}"
    )

_request_counter = 0


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _sse(payload: dict) -> bytes:
    """Format one SSE frame (`data: <json>\\n\\n`), as the OpenAI stream uses."""
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _should_fail() -> bool:
    return MOCK_FAIL_MODE != "none" and _request_counter > MOCK_FAIL_AFTER_REQ


def _chunk(cmpl_id: str, created: int, model: str, delta: dict,
           finish_reason: str | None = None) -> dict:
    return {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# ------------------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------------------

async def list_models(request: web.Request) -> web.Response:
    return web.json_response({
        "object": "list",
        "data": [{
            "id": SERVED_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mock-vllm",
        }],
    })


async def chat_completions(request: web.Request) -> web.StreamResponse:
    global _request_counter
    _request_counter += 1

    body = await request.json()
    max_tokens = int(body.get("max_tokens") or 128)
    model = body.get("model", SERVED_NAME)
    stream = bool(body.get("stream", False))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))

    # ---- failure injection ---------------------------------------------------
    if _should_fail():
        if MOCK_FAIL_MODE == "oom":
            return web.json_response(
                {"error": {
                    "message": "CUDA out of memory: KV cache full, cannot allocate",
                    "type": "engine_error", "code": 500,
                }},
                status=500,
            )
        if MOCK_FAIL_MODE == "http_500":
            return web.json_response(
                {"error": {"message": "internal server error",
                           "type": "engine_error", "code": 500}},
                status=500,
            )
        if MOCK_FAIL_MODE == "timeout":
            # Sleep longer than any reasonable client timeout.
            await asyncio.sleep(3600)
            return web.json_response({}, status=200)
        if MOCK_FAIL_MODE == "empty":
            # 200 OK, but emit zero content tokens (covers the empty-stream branch
            # in request_runner that produces failure_reason="other").
            stream = True
            max_tokens = 0

    # ---- non-streaming fast path --------------------------------------------
    if not stream:
        return web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok " * max(max_tokens, 1)},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": max_tokens,
                "total_tokens": max_tokens,
            },
        })

    # ---- streaming path ------------------------------------------------------
    resp = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)

    cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # Role-only opener: real OpenAI/vLLM streams send a role delta with empty
    # content first. The benchmark client correctly skips this when measuring
    # TTFT, so include it to exercise that code path.
    await resp.write(_sse(_chunk(cmpl_id, created, model, {"role": "assistant"})))

    # Simulate prefill before the first content token.
    if MOCK_TTFT_MS > 0 and max_tokens > 0:
        await asyncio.sleep(MOCK_TTFT_MS / 1000.0)

    # Stream content tokens, one per chunk.
    for i in range(max_tokens):
        if i > 0 and MOCK_TPOT_MS > 0:
            await asyncio.sleep(MOCK_TPOT_MS / 1000.0)
        await resp.write(_sse(_chunk(cmpl_id, created, model, {"content": "ok "})))

    # Stop chunk (empty delta, finish_reason="stop").
    await resp.write(_sse(_chunk(cmpl_id, created, model, {}, finish_reason="stop")))

    # Usage chunk -- the client passes stream_options={"include_usage": True}
    # and uses completion_tokens for the TPOT denominator.
    if include_usage:
        await resp.write(_sse({
            "id": cmpl_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": max_tokens,
                "total_tokens": max_tokens,
            },
        }))

    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


# ------------------------------------------------------------------------------
# App factory + entry point
# ------------------------------------------------------------------------------

def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


def main() -> None:
    print(f"[mock-vllm] listening on http://{HOST}:{PORT}")
    print(f"[mock-vllm] served_name = {SERVED_NAME}")
    print(f"[mock-vllm] ttft = {MOCK_TTFT_MS} ms   tpot = {MOCK_TPOT_MS} ms")
    print(f"[mock-vllm] fail_mode = {MOCK_FAIL_MODE}   fail_after_req = {MOCK_FAIL_AFTER_REQ}")
    web.run_app(make_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()

"""
Tests for the OOM-classification regex used by request_runner.

If this regex misses a real vLLM OOM string, the cell ends up labeled
http_error / other instead of oom, and the user can't tell why their sweep
hit a wall. False positives on unrelated 5xx are also bad: they upgrade
benign failures to OOM and falsely shrink the feasibility envelope.
"""
from __future__ import annotations

from benchmark.request_runner import _OOM_PATTERNS


class TestOOMPatternsPositive:
    def test_cuda_out_of_memory(self):
        assert _OOM_PATTERNS.search("CUDA out of memory") is not None

    def test_torch_oom_error(self):
        assert _OOM_PATTERNS.search(
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ..."
        ) is not None

    def test_kv_cache_full(self):
        assert _OOM_PATTERNS.search("KV cache full, cannot allocate") is not None

    def test_no_available_memory(self):
        assert _OOM_PATTERNS.search("no available memory for new request") is not None

    def test_engine_dead(self):
        assert _OOM_PATTERNS.search("Engine is dead") is not None

    def test_cannot_allocate(self):
        assert _OOM_PATTERNS.search("cannot allocate buffer") is not None

    def test_case_insensitive(self):
        assert _OOM_PATTERNS.search("OUT OF MEMORY") is not None
        assert _OOM_PATTERNS.search("Cuda Error: ...") is not None


class TestOOMPatternsNegative:
    def test_does_not_match_400(self):
        assert _OOM_PATTERNS.search("400: invalid request body") is None

    def test_does_not_match_connection_refused(self):
        assert _OOM_PATTERNS.search("connection refused") is None

    def test_does_not_match_model_not_found(self):
        assert _OOM_PATTERNS.search("model 'foo' not found") is None

    def test_does_not_match_generic_5xx(self):
        assert _OOM_PATTERNS.search("internal server error") is None

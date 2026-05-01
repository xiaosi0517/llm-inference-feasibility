"""
Smoke tests for YAML config loading. These guarantee that the configs shipped
in the repo never silently lose required fields.
"""
from __future__ import annotations

from pathlib import Path

from benchmark.sweep_benchmark import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_qwen14b_config_loads():
    cfg = load_config(str(REPO_ROOT / "configs" / "qwen14b_awq.yaml"))
    assert cfg.context_lengths == [1024, 4096, 8192, 16384]
    assert cfg.concurrency_levels == [1, 2, 4, 8]
    assert cfg.total_vram_mb == 24576
    assert cfg.feasible_max_mb < cfg.total_vram_mb
    assert cfg.min_success_rate == 1.0
    assert cfg.model_served_name == "qwen2.5-14b-awq"


def test_qwen32b_config_loads():
    cfg = load_config(str(REPO_ROOT / "configs" / "qwen32b_awq.yaml"))
    assert cfg.context_lengths == [1024, 2048, 4096, 8192]
    assert cfg.concurrency_levels == [1, 2, 4]
    assert cfg.feasible_max_mb < cfg.total_vram_mb
    assert cfg.request_timeout_s >= 120.0   # 32B prefill is slower; SLO is looser


def test_mock_config_loads_if_present():
    """The local-dev mock config is optional but should parse if it exists."""
    path = REPO_ROOT / "configs" / "mock.yaml"
    if not path.exists():
        return
    cfg = load_config(str(path))
    assert cfg.context_lengths
    assert cfg.concurrency_levels
    assert cfg.measured_bursts >= 1

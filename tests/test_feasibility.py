"""
Tests for the sweep_benchmark feasibility classifier and the percentile helper.
These are the two pieces of pure logic that decide what color every cell on
the heatmap turns. Wrong logic here -> silently wrong feasibility map.
"""
from __future__ import annotations

from benchmark.sweep_benchmark import SweepConfig, _pct, classify_feasibility


def make_cfg(**overrides) -> SweepConfig:
    base = dict(
        model_hf_repo="x", model_served_name="x",
        base_url="http://x", api_key="x",
        context_lengths=[1024], concurrency_levels=[1],
        max_output_tokens=128, warmup_bursts=0, measured_bursts=1,
        request_timeout_s=60.0,
        total_vram_mb=24576, feasible_max_mb=22528,
        p95_latency_ms_max=30000.0, min_success_rate=1.0,
        device_indices=[0], monitor_interval_s=0.1,
    )
    base.update(overrides)
    return SweepConfig(**base)


class TestClassifyFeasibility:
    def test_clean_run_under_thresholds_is_feasible(self):
        cfg = make_cfg()
        assert classify_feasibility(
            cfg, n_success=4, n_total=4,
            peak_mb=10000, p95_latency_ms=5000, failure_reason="ok",
        ) == "feasible"

    def test_oom_failure_is_infeasible(self):
        cfg = make_cfg()
        assert classify_feasibility(
            cfg, n_success=3, n_total=4,
            peak_mb=10000, p95_latency_ms=5000, failure_reason="oom",
        ) == "infeasible"

    def test_timeout_failure_is_infeasible(self):
        cfg = make_cfg()
        assert classify_feasibility(
            cfg, n_success=3, n_total=4,
            peak_mb=10000, p95_latency_ms=5000, failure_reason="timeout",
        ) == "infeasible"

    def test_peak_at_or_above_total_vram_is_infeasible(self):
        cfg = make_cfg()
        assert classify_feasibility(
            cfg, n_success=4, n_total=4,
            peak_mb=24576, p95_latency_ms=5000, failure_reason="ok",
        ) == "infeasible"

    def test_peak_in_marginal_band_is_marginal(self):
        cfg = make_cfg()
        assert classify_feasibility(
            cfg, n_success=4, n_total=4,
            peak_mb=23000, p95_latency_ms=5000, failure_reason="ok",
        ) == "marginal"

    def test_p95_over_slo_is_marginal(self):
        cfg = make_cfg()
        assert classify_feasibility(
            cfg, n_success=4, n_total=4,
            peak_mb=10000, p95_latency_ms=40000, failure_reason="ok",
        ) == "marginal"

    def test_no_gpu_data_falls_back_to_marginal(self):
        cfg = make_cfg()
        assert classify_feasibility(
            cfg, n_success=4, n_total=4,
            peak_mb=None, p95_latency_ms=5000, failure_reason="ok",
        ) == "marginal"

    def test_partial_success_below_floor_is_infeasible(self):
        cfg = make_cfg(min_success_rate=1.0)
        assert classify_feasibility(
            cfg, n_success=3, n_total=4,
            peak_mb=10000, p95_latency_ms=5000, failure_reason="ok",
        ) == "infeasible"

    def test_zero_total_is_infeasible(self):
        cfg = make_cfg()
        assert classify_feasibility(
            cfg, n_success=0, n_total=0,
            peak_mb=None, p95_latency_ms=None, failure_reason="ok",
        ) == "infeasible"

    def test_boundary_at_feasible_max_is_marginal(self):
        cfg = make_cfg(feasible_max_mb=22528)
        # Exactly at the ceiling -> marginal (per the spec: peak_mb >= feasible_max_mb).
        assert classify_feasibility(
            cfg, n_success=4, n_total=4,
            peak_mb=22528, p95_latency_ms=5000, failure_reason="ok",
        ) == "marginal"


class TestPercentile:
    def test_empty_returns_none(self):
        assert _pct([], 0.5) is None
        assert _pct([], 0.95) is None

    def test_single_value_any_quantile(self):
        assert _pct([42.0], 0.5) == 42.0
        assert _pct([42.0], 0.95) == 42.0
        assert _pct([42.0], 0.0) == 42.0

    def test_p50_and_p95_nearest_rank(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _pct(vals, 0.5) == 3.0
        assert _pct(vals, 0.95) == 5.0
        assert _pct(vals, 0.0) == 1.0

    def test_unsorted_input_is_sorted_first(self):
        assert _pct([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0

    def test_p95_under_small_n(self):
        # 3 samples is the typical measured_bursts*concurrency for conc=1.
        # nearest-rank p95 of [10, 20, 30] should pick the largest.
        assert _pct([10.0, 20.0, 30.0], 0.95) == 30.0

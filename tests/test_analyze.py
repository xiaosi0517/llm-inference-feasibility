"""
Tests for analyze_results: pivoting, formatting, plotting, and report writing.
The plotting tests just verify a non-empty PNG lands on disk; rendering
correctness is verified visually elsewhere.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; no display required for CI / dev

import pandas as pd

from benchmark.analyze_results import (
    _fmt,
    pivot,
    plot_feasibility_heatmap,
    plot_numeric_heatmap,
    write_decision_matrix,
    write_summary_report,
)


def _toy_df() -> pd.DataFrame:
    """4 cells: 2 ctx x 2 conc, mixed labels including one OOM."""
    return pd.DataFrame([
        dict(context_length=1024, concurrency=1, n_requests=3, n_success=3,
             success_rate=1.0, ttft_ms_p50=100.0, ttft_ms_p95=120.0,
             tpot_ms_p50=10.0, tpot_ms_p95=12.0,
             latency_ms_p50=500.0, latency_ms_p95=600.0,
             throughput_tps=80.0, gpu_mem_peak_mb=10000.0, gpu_mem_mean_mb=9500.0,
             feasibility="feasible", failure_reason="ok", burst_wall_time_s=4.5),
        dict(context_length=1024, concurrency=2, n_requests=6, n_success=6,
             success_rate=1.0, ttft_ms_p50=120.0, ttft_ms_p95=150.0,
             tpot_ms_p50=12.0, tpot_ms_p95=15.0,
             latency_ms_p50=700.0, latency_ms_p95=900.0,
             throughput_tps=140.0, gpu_mem_peak_mb=23000.0, gpu_mem_mean_mb=22000.0,
             feasibility="marginal", failure_reason="ok", burst_wall_time_s=5.5),
        dict(context_length=4096, concurrency=1, n_requests=3, n_success=3,
             success_rate=1.0, ttft_ms_p50=400.0, ttft_ms_p95=480.0,
             tpot_ms_p50=11.0, tpot_ms_p95=13.0,
             latency_ms_p50=1900.0, latency_ms_p95=2200.0,
             throughput_tps=70.0, gpu_mem_peak_mb=12000.0, gpu_mem_mean_mb=11000.0,
             feasibility="feasible", failure_reason="ok", burst_wall_time_s=6.5),
        dict(context_length=4096, concurrency=2, n_requests=6, n_success=2,
             success_rate=0.33, ttft_ms_p50=None, ttft_ms_p95=None,
             tpot_ms_p50=None, tpot_ms_p95=None,
             latency_ms_p50=None, latency_ms_p95=None,
             throughput_tps=None, gpu_mem_peak_mb=24600.0, gpu_mem_mean_mb=23000.0,
             feasibility="infeasible", failure_reason="oom", burst_wall_time_s=7.5),
    ])


class TestFmt:
    def test_none_renders_dash(self):
        assert _fmt(None) == "—"

    def test_nan_and_inf_render_dash(self):
        assert _fmt(float("nan")) == "—"
        assert _fmt(float("inf")) == "—"

    def test_number_uses_default_spec(self):
        assert _fmt(3.14) == "3.1"

    def test_custom_spec(self):
        assert _fmt(3.14159, "{:.2f}") == "3.14"
        assert _fmt(12345.0, "{:.0f}") == "12345"


class TestPivot:
    def test_pivot_shape_and_axis_order(self):
        grid = pivot(_toy_df(), "throughput_tps")
        # ctx descending (high at top), conc ascending (low on left)
        assert list(grid.columns) == [1, 2]
        assert list(grid.index) == [4096, 1024]

    def test_pivot_value_lookup(self):
        grid = pivot(_toy_df(), "throughput_tps")
        assert grid.loc[1024, 1] == 80.0
        assert grid.loc[1024, 2] == 140.0
        assert grid.loc[4096, 1] == 70.0


class TestPlots:
    def test_feasibility_heatmap_writes_png(self, tmp_path: Path):
        out = tmp_path / "feas.png"
        plot_feasibility_heatmap(_toy_df(), out)
        assert out.exists() and out.stat().st_size > 0

    def test_numeric_heatmap_writes_png(self, tmp_path: Path):
        out = tmp_path / "lat.png"
        plot_numeric_heatmap(
            _toy_df(), "latency_ms_p95", "p95 latency", out,
            log_scale=True, fmt="{:.0f}",
        )
        assert out.exists() and out.stat().st_size > 0


class TestReports:
    def test_decision_matrix_lists_every_cell(self, tmp_path: Path):
        out = tmp_path / "dm.md"
        write_decision_matrix(_toy_df(), out)
        text = out.read_text(encoding="utf-8")
        for ctx in [1024, 4096]:
            for conc in [1, 2]:
                assert f"| {ctx} | {conc} |" in text

    def test_summary_includes_best_throughput_and_oom_boundary(self, tmp_path: Path):
        out = tmp_path / "sum.md"
        write_summary_report(_toy_df(), out)
        text = out.read_text(encoding="utf-8")
        assert "Highest-throughput feasible config" in text
        assert "First OOM observed at ctx=4096" in text

    def test_summary_when_no_feasible_cells(self, tmp_path: Path):
        df = _toy_df().assign(feasibility="infeasible")
        out = tmp_path / "sum.md"
        write_summary_report(df, out)
        text = out.read_text(encoding="utf-8")
        assert "No feasible cells" in text

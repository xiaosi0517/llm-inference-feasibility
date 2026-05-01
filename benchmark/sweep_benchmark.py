"""
benchmark/sweep_benchmark.py

Drive the (context_length x concurrency) sweep against a running vLLM server.

For each cell:
  1. Build N unique prompts of the target token length (suffix-tagged so the
     server-side prefix cache doesn't trivialize the workload at high concurrency).
  2. Run a warmup burst (results discarded) so the KV allocator and CUDA graphs
     are warm before measurement.
  3. Run M measured bursts of N concurrent requests.
  4. Aggregate per-cell percentiles + throughput, sample peak VRAM via NVML,
     apply the 3-tier feasibility rule, and append a CSV row immediately
     (so a crash mid-sweep doesn't lose completed cells).

Output:
    benchmark_results.csv -- one row per (ctx, conc) cell. The analyzer
    (benchmark/analyze_results.py) consumes this CSV to produce the
    feasibility heatmap, decision matrix, and summary report.

Usage:
    python -m benchmark.sweep_benchmark --config configs/qwen14b_awq.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from benchmark.gpu_monitor import GpuMonitor
from benchmark.request_runner import (
    RequestResult,
    RunnerConfig,
    make_client,
    send_concurrent,
)


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

@dataclass
class SweepConfig:
    model_hf_repo: str
    model_served_name: str
    base_url: str
    api_key: str

    context_lengths: list[int]
    concurrency_levels: list[int]
    max_output_tokens: int
    warmup_bursts: int
    measured_bursts: int
    request_timeout_s: float

    total_vram_mb: int
    feasible_max_mb: int
    p95_latency_ms_max: float
    min_success_rate: float

    device_indices: list[int]
    monitor_interval_s: float


def load_config(path: str) -> SweepConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SweepConfig(
        model_hf_repo=raw["model"]["hf_repo"],
        model_served_name=raw["model"]["served_name"],
        base_url=raw["server"]["base_url"],
        api_key=raw["server"].get("api_key", "EMPTY"),
        context_lengths=raw["sweep"]["context_lengths"],
        concurrency_levels=raw["sweep"]["concurrency_levels"],
        max_output_tokens=raw["sweep"]["max_output_tokens"],
        warmup_bursts=raw["sweep"].get("warmup_bursts", 1),
        measured_bursts=raw["sweep"].get("measured_bursts", 3),
        request_timeout_s=raw["sweep"].get("request_timeout_s", 120.0),
        total_vram_mb=raw["feasibility"]["total_vram_mb"],
        feasible_max_mb=raw["feasibility"]["feasible_max_mb"],
        p95_latency_ms_max=raw["feasibility"]["p95_latency_ms_max"],
        min_success_rate=raw["feasibility"]["min_success_rate"],
        device_indices=raw["monitor"]["device_indices"],
        monitor_interval_s=raw["monitor"]["interval_s"],
    )


# ------------------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------------------

def build_prompts(model_name: str, target_tokens: int, count: int) -> list[str]:
    """
    Return `count` prompts whose user-content tokenizes to ~target_tokens.

    We round-trip filler text through the model's HF tokenizer to get an exact
    token count, then append a unique short suffix to each prompt so vLLM's
    automatic prefix caching doesn't make every concurrent request hit the
    same cached prefill (which would understate TTFT under load).

    The chat template the server applies adds ~10-20 tokens of overhead; we
    log the *requested* context_length, not the inflated server-side count,
    so cells are comparable across configs.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_filler = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tok(base_filler, add_special_tokens=False)["input_ids"]
    while len(ids) < target_tokens:
        base_filler = base_filler * 2
        ids = tok(base_filler, add_special_tokens=False)["input_ids"]
    ids = ids[:target_tokens]
    base_text = tok.decode(ids, skip_special_tokens=True)
    return [f"{base_text}\n[req={i}]" for i in range(count)]


# ------------------------------------------------------------------------------
# Per-cell metrics
# ------------------------------------------------------------------------------

@dataclass
class CellMetrics:
    context_length: int
    concurrency: int
    n_requests: int
    n_success: int
    success_rate: float
    ttft_ms_p50: Optional[float]
    ttft_ms_p95: Optional[float]
    tpot_ms_p50: Optional[float]
    tpot_ms_p95: Optional[float]
    latency_ms_p50: Optional[float]
    latency_ms_p95: Optional[float]
    throughput_tps: Optional[float]      # aggregate output tokens / wall-time of measured bursts
    gpu_mem_peak_mb: Optional[float]
    gpu_mem_mean_mb: Optional[float]
    feasibility: str                     # feasible | marginal | infeasible
    failure_reason: str                  # ok | timeout | oom | http_error | other
    burst_wall_time_s: float


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    # nearest-rank percentile -- robust for small N (typical: 3-24 samples per cell)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def classify_feasibility(
    cfg: SweepConfig,
    n_success: int,
    n_total: int,
    peak_mb: Optional[float],
    p95_latency_ms: Optional[float],
    failure_reason: str,
) -> str:
    """
    3-tier rule from the spec.

      infeasible: any failure (oom/timeout/http_error/other), or success_rate below floor,
                  or peak_vram >= total_vram_mb
      marginal:   no errors, peak_vram in [feasible_max_mb, total_vram_mb),
                  OR p95 latency exceeds SLO,
                  OR no GPU monitor data available (be conservative)
      feasible:   no errors, peak_vram < feasible_max_mb, p95 latency within SLO
    """
    if n_total == 0:
        return "infeasible"
    if failure_reason != "ok" and n_success < n_total:
        return "infeasible"
    success_rate = n_success / n_total
    if success_rate < cfg.min_success_rate:
        return "infeasible"
    if peak_mb is None:
        return "marginal"
    if peak_mb >= cfg.total_vram_mb:
        return "infeasible"
    if peak_mb >= cfg.feasible_max_mb:
        return "marginal"
    if p95_latency_ms is not None and p95_latency_ms > cfg.p95_latency_ms_max:
        return "marginal"
    return "feasible"


# ------------------------------------------------------------------------------
# One cell run
# ------------------------------------------------------------------------------

async def run_cell(
    cfg: SweepConfig,
    runner_cfg: RunnerConfig,
    context_length: int,
    concurrency: int,
) -> CellMetrics:
    total_bursts = cfg.warmup_bursts + cfg.measured_bursts
    prompts = build_prompts(
        cfg.model_hf_repo, context_length, concurrency * total_bursts
    )

    client = make_client(runner_cfg)
    measured_results: list[RequestResult] = []
    burst_wall_time_s_total = 0.0
    failure_reason = "ok"

    monitor = GpuMonitor(
        device_indices=cfg.device_indices, interval_s=cfg.monitor_interval_s
    )
    monitor.start()

    try:
        # ---- warmup ---------------------------------------------------------
        for b in range(cfg.warmup_bursts):
            batch = prompts[b * concurrency : (b + 1) * concurrency]
            await send_concurrent(client, runner_cfg, batch)

        # ---- measured -------------------------------------------------------
        offset = cfg.warmup_bursts * concurrency
        for b in range(cfg.measured_bursts):
            batch = prompts[offset + b * concurrency : offset + (b + 1) * concurrency]
            t_burst = time.perf_counter()
            results = await send_concurrent(client, runner_cfg, batch)
            burst_wall_time_s_total += time.perf_counter() - t_burst
            measured_results.extend(results)

            # First non-ok reason wins for the cell label (most informative
            # for diagnosing the failure mode).
            if failure_reason == "ok":
                for r in results:
                    if not r.ok:
                        failure_reason = r.failure_reason
                        break
    finally:
        await client.close()
        monitor.stop()

    # ---- aggregate ---------------------------------------------------------
    n_total = len(measured_results)
    successes = [r for r in measured_results if r.ok]
    n_success = len(successes)

    ttft = [r.ttft_ms for r in successes if r.ttft_ms is not None]
    lat = [r.latency_ms for r in successes if r.latency_ms is not None]
    tpot = [r.tpot_ms for r in successes if r.tpot_ms is not None]

    total_out_tokens = sum((r.output_tokens or 0) for r in successes)
    throughput = (
        total_out_tokens / burst_wall_time_s_total
        if burst_wall_time_s_total > 0 and total_out_tokens > 0
        else None
    )

    p95_lat = _pct(lat, 0.95)
    feasibility = classify_feasibility(
        cfg, n_success, n_total, monitor.peak_mb, p95_lat, failure_reason
    )

    return CellMetrics(
        context_length=context_length,
        concurrency=concurrency,
        n_requests=n_total,
        n_success=n_success,
        success_rate=(n_success / n_total) if n_total else 0.0,
        ttft_ms_p50=_pct(ttft, 0.5),
        ttft_ms_p95=_pct(ttft, 0.95),
        tpot_ms_p50=_pct(tpot, 0.5),
        tpot_ms_p95=_pct(tpot, 0.95),
        latency_ms_p50=_pct(lat, 0.5),
        latency_ms_p95=p95_lat,
        throughput_tps=throughput,
        gpu_mem_peak_mb=monitor.peak_mb,
        gpu_mem_mean_mb=monitor.mean_mb,
        feasibility=feasibility,
        failure_reason=failure_reason,
        burst_wall_time_s=burst_wall_time_s_total,
    )


# ------------------------------------------------------------------------------
# Sweep driver
# ------------------------------------------------------------------------------

async def run_sweep(cfg: SweepConfig, out_csv: Path) -> list[CellMetrics]:
    runner_cfg = RunnerConfig(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model_served_name,
        max_output_tokens=cfg.max_output_tokens,
        request_timeout_s=cfg.request_timeout_s,
    )

    cells: list[CellMetrics] = []
    fieldnames = list(CellMetrics.__dataclass_fields__.keys())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Walk in (ctx, conc) order so a partial CSV is still readable as
        # "first few context lengths fully measured".
        grid = [(c, n) for c in cfg.context_lengths for n in cfg.concurrency_levels]
        for ctx, conc in tqdm(grid, desc="sweep"):
            try:
                m = await run_cell(cfg, runner_cfg, ctx, conc)
            except Exception as exc:  # noqa: BLE001
                # Driver-level failure (e.g. tokenizer download failed). Record
                # the cell as infeasible with the error so the sweep continues.
                m = CellMetrics(
                    context_length=ctx, concurrency=conc, n_requests=0, n_success=0,
                    success_rate=0.0, ttft_ms_p50=None, ttft_ms_p95=None,
                    tpot_ms_p50=None, tpot_ms_p95=None, latency_ms_p50=None,
                    latency_ms_p95=None, throughput_tps=None,
                    gpu_mem_peak_mb=None, gpu_mem_mean_mb=None,
                    feasibility="infeasible",
                    failure_reason=f"driver_error: {exc!s}"[:200],
                    burst_wall_time_s=0.0,
                )
            cells.append(m)
            writer.writerow(asdict(m))
            f.flush()  # crash-resilience: each completed cell is durable

    return cells


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qwen14b_awq.yaml",
                   help="Path to sweep YAML config.")
    p.add_argument("--out-csv", default="results/benchmark_results.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"[sweep] model={cfg.model_served_name} grid="
          f"{len(cfg.context_lengths)}x{len(cfg.concurrency_levels)}")
    cells = asyncio.run(run_sweep(cfg, Path(args.out_csv)))

    # Compact summary so `run_sweep.sh` users see something useful in the log.
    counts = {"feasible": 0, "marginal": 0, "infeasible": 0}
    for c in cells:
        counts[c.feasibility] = counts.get(c.feasibility, 0) + 1
    print(f"[sweep] done: {counts}  -> {args.out_csv}")


if __name__ == "__main__":
    main()

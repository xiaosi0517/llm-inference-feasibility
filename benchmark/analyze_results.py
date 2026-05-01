"""
benchmark/analyze_results.py

Post-process a sweep CSV and produce the MVP deliverables:
  - results/feasibility_heatmap.png   (Feasible / Marginal / Infeasible grid)
  - results/decision_matrix.md        (per (ctx, conc) row with metrics + label)
  - results/summary_report.md         (best-feasible config, OOM boundary)

The latency / throughput numbers are surfaced in `decision_matrix.md`; the
CSV remains the source of truth if you want to plot something specific later.
`plot_numeric_heatmap` is kept as available infrastructure but is not invoked
by the default pipeline.

Usage:
    python -m benchmark.analyze_results results/benchmark_results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm


# ------------------------------------------------------------------------------
# Loading + pivoting
# ------------------------------------------------------------------------------

FEAS_ORDER = ["feasible", "marginal", "infeasible"]
FEAS_COLORS = {"feasible": "#2ecc71", "marginal": "#f1c40f", "infeasible": "#e74c3c"}
FEAS_TO_INT = {"feasible": 0, "marginal": 1, "infeasible": 2}


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure rows/cols sort numerically, not lexicographically.
    df["context_length"] = df["context_length"].astype(int)
    df["concurrency"] = df["concurrency"].astype(int)
    return df


def pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot to context_length (rows) x concurrency (cols)."""
    return (
        df.pivot(index="context_length", columns="concurrency", values=value_col)
          .sort_index(ascending=False)         # high ctx at top -> reads naturally
          .sort_index(axis=1, ascending=True)  # low conc on left
    )


# ------------------------------------------------------------------------------
# Heatmaps
# ------------------------------------------------------------------------------

def plot_feasibility_heatmap(df: pd.DataFrame, out: Path) -> None:
    """3-color discrete heatmap: green / yellow / red."""
    grid = pivot(df.assign(_f=df["feasibility"].map(FEAS_TO_INT)), "_f")

    cmap = ListedColormap([FEAS_COLORS[k] for k in FEAS_ORDER])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(1.3 * len(grid.columns) + 2, 0.7 * len(grid) + 2))
    im = ax.imshow(grid.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(grid.columns)), [f"c={c}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{c}" for c in grid.index])
    ax.set_xlabel("concurrency")
    ax.set_ylabel("context length")
    ax.set_title("Feasibility")

    # Annotate each cell with its label so the figure is self-contained.
    for i, ctx in enumerate(grid.index):
        for j, conc in enumerate(grid.columns):
            v = grid.values[i, j]
            if pd.isna(v):
                continue
            label = FEAS_ORDER[int(v)]
            ax.text(j, i, label, ha="center", va="center",
                    color="black", fontsize=8)

    cbar = fig.colorbar(im, ticks=[0, 1, 2])
    cbar.set_ticklabels(FEAS_ORDER)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_numeric_heatmap(
    df: pd.DataFrame, value_col: str, title: str, out: Path,
    log_scale: bool = False, cmap: str = "viridis", fmt: str = "{:.0f}",
) -> None:
    grid = pivot(df, value_col).astype(float)

    fig, ax = plt.subplots(figsize=(1.3 * len(grid.columns) + 2, 0.7 * len(grid) + 2))

    if log_scale and (grid.values > 0).any():
        # ignore NaN/zeros for the color norm
        finite = grid.values[np.isfinite(grid.values) & (grid.values > 0)]
        norm = LogNorm(vmin=finite.min(), vmax=finite.max()) if finite.size else None
        im = ax.imshow(grid.values, cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(grid.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(grid.columns)), [f"c={c}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{c}" for c in grid.index])
    ax.set_xlabel("concurrency")
    ax.set_ylabel("context length")
    ax.set_title(title)

    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            v = grid.values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", color="white", fontsize=8)
            else:
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white", fontsize=8)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Markdown reports
# ------------------------------------------------------------------------------

def _fmt(v, spec: str = "{:.1f}") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


def write_decision_matrix(df: pd.DataFrame, out: Path) -> None:
    """One row per (ctx, conc) cell with metrics + label + recommendation."""
    rows = ["# Decision matrix",
            "",
            "| ctx | conc | feas | TTFT p95 (ms) | TPOT p95 (ms) | latency p95 (ms) | tput (tok/s) | peak VRAM (MB) | reason |",
            "|---:|---:|:---|---:|---:|---:|---:|---:|:---|"]
    for _, r in df.sort_values(["context_length", "concurrency"]).iterrows():
        rows.append(
            f"| {int(r.context_length)} | {int(r.concurrency)} | {r.feasibility} | "
            f"{_fmt(r.ttft_ms_p95)} | {_fmt(r.tpot_ms_p95)} | {_fmt(r.latency_ms_p95)} | "
            f"{_fmt(r.throughput_tps, '{:.1f}')} | {_fmt(r.gpu_mem_peak_mb, '{:.0f}')} | "
            f"{r.failure_reason} |"
        )
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_summary_report(df: pd.DataFrame, out: Path) -> None:
    """Headline numbers + the 'what should I deploy with' answer."""
    n_total = len(df)
    by_label = df.groupby("feasibility").size().to_dict()

    feasible = df[df.feasibility == "feasible"]
    best_throughput = (
        feasible.sort_values("throughput_tps", ascending=False).iloc[0]
        if not feasible.empty else None
    )
    largest_feasible_ctx = (
        feasible.sort_values(["context_length", "concurrency"], ascending=False).iloc[0]
        if not feasible.empty else None
    )

    # OOM boundary: smallest (ctx, conc) that infeasibled with reason=="oom"
    oom = df[df.failure_reason == "oom"]
    first_oom = (
        oom.sort_values(["context_length", "concurrency"]).iloc[0]
        if not oom.empty else None
    )

    lines = [
        "# Summary report",
        "",
        f"- Total cells measured: **{n_total}**",
        f"- Feasibility breakdown: "
        f"feasible={by_label.get('feasible', 0)}, "
        f"marginal={by_label.get('marginal', 0)}, "
        f"infeasible={by_label.get('infeasible', 0)}",
        "",
        "## Recommended deployments",
        "",
    ]
    if best_throughput is not None:
        lines.append(
            f"- **Highest-throughput feasible config:** "
            f"ctx={int(best_throughput.context_length)}, "
            f"conc={int(best_throughput.concurrency)} "
            f"→ {_fmt(best_throughput.throughput_tps, '{:.1f}')} tok/s, "
            f"peak VRAM {_fmt(best_throughput.gpu_mem_peak_mb, '{:.0f}')} MB."
        )
    if largest_feasible_ctx is not None:
        lines.append(
            f"- **Largest feasible context window:** "
            f"ctx={int(largest_feasible_ctx.context_length)} at "
            f"conc={int(largest_feasible_ctx.concurrency)} "
            f"(p95 latency {_fmt(largest_feasible_ctx.latency_ms_p95)} ms)."
        )
    if best_throughput is None and largest_feasible_ctx is None:
        lines.append("- No feasible cells. Try a smaller model, FP8 KV cache, "
                     "or reduce max-model-len / max-num-seqs.")

    lines += ["", "## OOM boundary", ""]
    if first_oom is not None:
        lines.append(
            f"- First OOM observed at ctx={int(first_oom.context_length)}, "
            f"conc={int(first_oom.concurrency)} "
            f"(peak VRAM {_fmt(first_oom.gpu_mem_peak_mb, '{:.0f}')} MB). "
            f"See `docs/decision_framework.md` for mitigations."
        )
    else:
        lines.append("- No OOMs recorded in this sweep.")

    lines += ["", "See `decision_matrix.md` for the per-cell breakdown "
                  "and `feasibility_heatmap.png` for the visual."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="results/benchmark_results.csv")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)

    plot_feasibility_heatmap(df, out_dir / "feasibility_heatmap.png")
    write_decision_matrix(df, out_dir / "decision_matrix.md")
    write_summary_report(df, out_dir / "summary_report.md")

    print(f"[analyze] wrote feasibility heatmap + reports to {out_dir}/")


if __name__ == "__main__":
    main()

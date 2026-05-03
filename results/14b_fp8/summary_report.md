# Summary report

- Total cells measured: **16**
- Feasibility breakdown: feasible=6, marginal=2, infeasible=8

## Recommended deployments

- **Highest-throughput feasible config:** ctx=1024, conc=8 → 314.0 tok/s, peak VRAM 20806 MB.
- **Largest feasible context window:** ctx=4096 at conc=2 (p95 latency 3258.6 ms).

## OOM boundary

- No OOMs recorded in this sweep.

See `decision_matrix.md` for the per-cell breakdown and `feasibility_heatmap.png` for the visual.

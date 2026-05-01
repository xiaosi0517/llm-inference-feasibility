# Summary report

- Total cells measured: **12**
- Feasibility breakdown: feasible=1, marginal=5, infeasible=6

## Recommended deployments

- **Highest-throughput feasible config:** ctx=1024, conc=1 → 37.8 tok/s, peak VRAM 22386 MB.
- **Largest feasible context window:** ctx=1024 at conc=1 (p95 latency 3387.4 ms).

## OOM boundary

- No OOMs recorded in this sweep.

See `decision_matrix.md` for the per-cell breakdown and `feasibility_heatmap.png` for the visual.

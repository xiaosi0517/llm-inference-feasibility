"""
Invariants for GpuMonitor that hold whether or not a real GPU/NVML is present.
The sweep driver depends on these contracts to keep running on CPU-only hosts
(local dev) and to label cells "marginal" instead of crashing.
"""
from __future__ import annotations

from benchmark.gpu_monitor import GpuMonitor


def test_lifecycle_does_not_raise_without_gpu():
    """start() then stop() must complete without exceptions on any host."""
    mon = GpuMonitor(device_indices=[0], interval_s=0.05)
    mon.start()
    mon.stop()


def test_empty_state_metrics_are_none():
    """Before any sample is collected, all metrics return None."""
    mon = GpuMonitor(device_indices=[0], interval_s=0.05)
    assert mon.peak_mb is None
    assert mon.mean_mb is None
    assert mon.final_mb is None
    assert mon.available is False


def test_context_manager_protocol():
    """`with GpuMonitor(...) as m:` is the documented usage; must not raise."""
    with GpuMonitor(device_indices=[0], interval_s=0.05) as mon:
        pass
    # After context exit the monitor is stopped and metrics are queryable.
    _ = mon.peak_mb
    _ = mon.mean_mb

"""
benchmark/gpu_monitor.py

Background sampler for GPU memory usage via NVML. Wrap one sweep cell:

    with GpuMonitor(device_indices=[0], interval_s=0.1) as mon:
        await run_one_cell(...)
    print(mon.peak_mb, mon.mean_mb, mon.samples)

Why a thread (not asyncio):
  Under high concurrency the event loop is busy draining N streaming
  responses; an asyncio.sleep(0.1) sampler goes jittery and misses peaks
  precisely when memory pressure matters. A daemon thread polls NVML
  independently of the loop.

Why NVML (not nvidia-smi):
  Shelling out forks a process per sample (~30-80 ms) and perturbs the GPU.
  pynvml calls the same library nvidia-smi uses, in-process, in <1 ms.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import pynvml  # type: ignore
    _HAS_NVML = True
except Exception:  # noqa: BLE001
    _HAS_NVML = False


@dataclass
class GpuSnapshot:
    """One reading across all monitored devices, summed."""
    t: float                      # perf_counter timestamp
    used_mb: float                # total used MiB across devices
    per_device_mb: list[float] = field(default_factory=list)


class GpuMonitor:
    """
    Samples GPU memory in a background thread. Use as a context manager.

    Args:
        device_indices: which CUDA devices to monitor. For TP=1 leave as [0].
                        For TP=N pass [0, 1, ..., N-1]; peak/mean are summed
                        across them so a single number characterizes the cell.
        interval_s: sampling period. 0.1 s (10 Hz) is the sweet spot:
                    fast enough to catch prefill spikes (~200 ms windows),
                    slow enough to not load the GIL.
    """

    def __init__(
        self,
        device_indices: list[int] | None = None,
        interval_s: float = 0.1,
    ) -> None:
        self.device_indices = device_indices if device_indices is not None else [0]
        self.interval_s = interval_s

        self._handles: list = []
        self._samples: list[GpuSnapshot] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._initialized = False
        self._init_error: Optional[str] = None

    # --- lifecycle ------------------------------------------------------------

    def __enter__(self) -> "GpuMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if not _HAS_NVML:
            self._init_error = "pynvml not installed"
            return
        try:
            pynvml.nvmlInit()
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.device_indices
            ]
            self._initialized = True
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"nvml init failed: {exc}"
            return

        self._stop_event.clear()
        self._samples = []
        self._thread = threading.Thread(
            target=self._run, name="gpu-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            # Give the loop one full interval + a small grace to exit cleanly.
            self._thread.join(timeout=self.interval_s + 1.0)
            self._thread = None
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            self._initialized = False

    # --- worker ---------------------------------------------------------------

    def _run(self) -> None:
        # Tight loop with sleep on a stoppable event so stop() returns quickly.
        while not self._stop_event.is_set():
            try:
                per_dev: list[float] = []
                for h in self._handles:
                    info = pynvml.nvmlDeviceGetMemoryInfo(h)
                    per_dev.append(info.used / (1024 * 1024))  # bytes -> MiB
                self._samples.append(
                    GpuSnapshot(t=time.perf_counter(),
                                used_mb=sum(per_dev),
                                per_device_mb=per_dev)
                )
            except Exception as exc:  # noqa: BLE001
                # Don't kill the thread on a single bad read; record and continue.
                self._init_error = f"sample failed: {exc}"
            self._stop_event.wait(self.interval_s)

    # --- accessors (safe to call after stop()) --------------------------------

    @property
    def samples(self) -> list[GpuSnapshot]:
        return list(self._samples)

    @property
    def available(self) -> bool:
        """True if at least one valid sample was collected."""
        return len(self._samples) > 0

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    @property
    def peak_mb(self) -> Optional[float]:
        if not self._samples:
            return None
        return max(s.used_mb for s in self._samples)

    @property
    def mean_mb(self) -> Optional[float]:
        if not self._samples:
            return None
        return sum(s.used_mb for s in self._samples) / len(self._samples)

    @property
    def final_mb(self) -> Optional[float]:
        if not self._samples:
            return None
        return self._samples[-1].used_mb


# ------------------------------------------------------------------------------
# Smoke test: python -m benchmark.gpu_monitor
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    with GpuMonitor(device_indices=[0], interval_s=0.1) as mon:
        time.sleep(2.0)
    if mon.init_error:
        print(f"NVML unavailable: {mon.init_error}")
    print(f"samples       : {len(mon.samples)}")
    print(f"peak_mb       : {mon.peak_mb}")
    print(f"mean_mb       : {mon.mean_mb}")
    print(f"final_mb      : {mon.final_mb}")

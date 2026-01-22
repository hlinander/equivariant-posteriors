"""
GPU monitoring with integration into analytics metrics.

Tracks power (Watts) and memory usage in a background thread.
Call record_metrics(step) from training loop to insert at specific steps.
"""
import threading
import time
from collections import deque
from typing import Optional

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMonitor:
    """
    Monitor GPU power and memory usage in a background thread.

    Usage:
        monitor = GPUMonitor(model_id, run_id)
        monitor.start()

        for step in range(num_steps):
            # ... training step ...
            monitor.record_metrics(step)  # Insert GPU metrics at this step

        monitor.stop()
    """

    def __init__(
        self,
        model_id: int,
        run_id: int,
        sample_interval: float = 1.0,
        window_size: int = 30,
    ):
        """
        Args:
            model_id: Model ID for metrics
            run_id: Run ID for metrics
            sample_interval: How often to sample GPU stats (seconds)
            window_size: Number of samples for rolling average
        """
        if not PYNVML_AVAILABLE:
            print("[gpu_monitor] pynvml not available, GPU monitoring disabled")
            self.enabled = False
            return

        self.model_id = model_id
        self.run_id = run_id
        self.sample_interval = sample_interval
        self.window_size = window_size

        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
            self.history = {i: deque(maxlen=window_size) for i in range(self.device_count)}
            self.enabled = True
            print(f"[gpu_monitor] Initialized with {self.device_count} GPU(s)")
        except Exception as e:
            print(f"[gpu_monitor] Failed to initialize NVML: {e}")
            self.enabled = False

    def _sample(self):
        """Sample current GPU stats"""
        samples = {}
        for i, handle in enumerate(self.handles):
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                samples[i] = {
                    "power_w": power_mw / 1000.0,
                    "mem_used_mb": mem_info.used / (1024 * 1024),
                    "mem_total_mb": mem_info.total / (1024 * 1024),
                    "util_pct": util.gpu,
                }
            except Exception:
                pass

        with self._lock:
            for i, sample in samples.items():
                self.history[i].append(sample)

        return samples

    def _monitor_loop(self):
        """Background sampling loop"""
        while not self._stop:
            self._sample()
            time.sleep(self.sample_interval)

    def start(self):
        """Start background sampling thread"""
        if not self.enabled:
            return self

        self._stop = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[gpu_monitor] Started (sample interval: {self.sample_interval}s)")
        return self

    def stop(self):
        """Stop sampling thread"""
        if not self.enabled or self._thread is None:
            return

        self._stop = True
        self._thread.join(timeout=5.0)
        print("[gpu_monitor] Stopped")

    def get_averages(self) -> dict:
        """Get rolling average stats for all GPUs"""
        if not self.enabled:
            return {}

        with self._lock:
            averages = {}
            for i, hist in self.history.items():
                if not hist:
                    continue
                averages[i] = {
                    "power_w": sum(s["power_w"] for s in hist) / len(hist),
                    "mem_used_mb": sum(s["mem_used_mb"] for s in hist) / len(hist),
                    "mem_total_mb": hist[-1]["mem_total_mb"] if hist else 0,
                    "util_pct": sum(s["util_pct"] for s in hist) / len(hist),
                }
        return averages

    def record_metrics(self, step: int):
        """
        Insert current averaged GPU metrics at the given training step.

        Call this from your training loop to correlate GPU metrics with training progress.
        """
        if not self.enabled:
            return

        import lib.render_duck as duck

        averages = self.get_averages()
        for gpu_idx, stats in averages.items():
            prefix = f"gpu{gpu_idx}" if self.device_count > 1 else "gpu"

            duck.insert_train_step_metric(
                self.model_id, self.run_id,
                f"{prefix}/power_w", step, stats["power_w"]
            )
            duck.insert_train_step_metric(
                self.model_id, self.run_id,
                f"{prefix}/mem_used_mb", step, stats["mem_used_mb"]
            )
            duck.insert_train_step_metric(
                self.model_id, self.run_id,
                f"{prefix}/mem_pct", step,
                (stats["mem_used_mb"] / stats["mem_total_mb"] * 100) if stats["mem_total_mb"] > 0 else 0
            )
            duck.insert_train_step_metric(
                self.model_id, self.run_id,
                f"{prefix}/util_pct", step, stats["util_pct"]
            )

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

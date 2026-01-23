"""
GPU monitoring with integration into analytics metrics.

Tracks power (Watts) and memory usage in a background thread.
Call record_metrics(step) from training loop - metrics are recorded at most
every record_interval seconds (default 10s) to avoid excessive logging.
"""
import threading
import time
from typing import Optional

from lib.log import log
from lib.train_dataclasses import GPUMonitorConfig

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMonitor:
    """
    Monitor GPU power and memory usage in a background thread.

    Samples GPU stats continuously and records averaged metrics at most every
    record_interval seconds (default 10s) when record_metrics() is called.
    Averages all samples collected since the last recording.

    Usage:
        from lib.train_dataclasses import GPUMonitorConfig
        config = GPUMonitorConfig()
        monitor = GPUMonitor(model_id, run_id, config)
        monitor.start()

        for step in range(num_steps):
            # ... training step ...
            monitor.record_metrics(step)  # Records only if 10s+ have passed

        monitor.stop()
    """

    def __init__(
        self,
        model_id: int,
        run_id: int,
        config: GPUMonitorConfig = None,
    ):
        """
        Args:
            model_id: Model ID for metrics
            run_id: Run ID for metrics
            config: GPU monitoring configuration
        """
        if config is None:
            config = GPUMonitorConfig()

        if not config.enabled:
            self.enabled = False
            return

        if not PYNVML_AVAILABLE:
            log("gpu_monitor", "pynvml not available, GPU monitoring disabled")
            self.enabled = False
            return

        self.model_id = model_id
        self.run_id = run_id
        self.sample_interval = config.sample_interval
        self.record_interval = config.record_interval
        self._last_record_time = 0.0

        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
            self.samples = {i: [] for i in range(self.device_count)}
            self.enabled = True
            log("gpu_monitor", f"Initialized with {self.device_count} GPU(s)")
        except Exception as e:
            log("gpu_monitor", f"Failed to initialize NVML: {e}")
            self.enabled = False

    def _sample(self):
        """Sample current GPU stats"""
        for i, handle in enumerate(self.handles):
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                sample = {
                    "power_w": power_mw / 1000.0,
                    "mem_used_mb": mem_info.used / (1024 * 1024),
                    "mem_total_mb": mem_info.total / (1024 * 1024),
                    "util_pct": util.gpu,
                }
                with self._lock:
                    self.samples[i].append(sample)
            except Exception:
                pass

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
        log("gpu_monitor", f"Started (sample: {self.sample_interval}s, record: {self.record_interval}s)")
        return self

    def stop(self):
        """Stop sampling thread"""
        if not self.enabled or self._thread is None:
            return

        self._stop = True
        self._thread.join(timeout=5.0)
        log("gpu_monitor", "Stopped")

    def _get_averages_and_clear(self) -> dict:
        """Get average stats for all GPUs and clear samples"""
        if not self.enabled:
            return {}

        with self._lock:
            averages = {}
            for i, samples in self.samples.items():
                if not samples:
                    continue
                averages[i] = {
                    "power_w": sum(s["power_w"] for s in samples) / len(samples),
                    "mem_used_mb": sum(s["mem_used_mb"] for s in samples) / len(samples),
                    "mem_total_mb": samples[-1]["mem_total_mb"],
                    "util_pct": sum(s["util_pct"] for s in samples) / len(samples),
                }
                self.samples[i] = []
        return averages

    def record_metrics(self, step: int):
        """
        Insert current averaged GPU metrics at the given training step.

        Call this from your training loop to correlate GPU metrics with training progress.
        Only records if record_interval seconds have passed since last recording.
        Averages all samples collected since the last recording.
        """
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_record_time < self.record_interval:
            return
        self._last_record_time = now

        import lib.render_duck as duck

        averages = self._get_averages_and_clear()
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

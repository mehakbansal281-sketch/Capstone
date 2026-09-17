import time
import os
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger("PerformanceMonitor")

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.response_times_ms = []
        self.inference_times_ms = []

    def record_request(self, duration_ms: float, is_error: bool = False):
        self.total_requests += 1
        if is_error:
            self.total_errors += 1
        self.response_times_ms.append(duration_ms)
        # Keep last 1000 records
        if len(self.response_times_ms) > 1000:
            self.response_times_ms.pop(0)

    def record_inference(self, duration_ms: float):
        self.inference_times_ms.append(duration_ms)
        if len(self.inference_times_ms) > 1000:
            self.inference_times_ms.pop(0)

    def get_metrics_summary(self) -> Dict[str, Any]:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        memory_mb = round(mem_info.rss / (1024 * 1024), 2)
        cpu_percent = process.cpu_percent(interval=None)
        uptime_seconds = round(time.time() - self.start_time, 2)

        avg_resp_time = (
            round(sum(self.response_times_ms) / len(self.response_times_ms), 2)
            if self.response_times_ms else 0.0
        )
        avg_inf_time = (
            round(sum(self.inference_times_ms) / len(self.inference_times_ms), 2)
            if self.inference_times_ms else 0.0
        )

        return {
            "uptime_seconds": uptime_seconds,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_response_time_ms": avg_resp_time,
            "avg_inference_time_ms": avg_inf_time,
            "system_metrics": {
                "memory_usage_mb": memory_mb,
                "cpu_usage_percent": cpu_percent
            }
        }

# Global monitor instance
monitor = PerformanceMonitor()

# timing_utils.py
from __future__ import annotations
import time
from contextlib import contextmanager

class Timings:
    def __init__(self):
        self.data = {}

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.data[name] = self.data.get(name, 0.0) + dt

    def reset(self):
        self.data.clear()

    def summary_str(self, order=None):
        if order is None:
            order = sorted(self.data.keys())
        parts = [f"{k}={self.data.get(k,0.0):.2f}s" for k in order if k in self.data]
        total = sum(self.data.values())
        return f"total={total:.2f}s | " + " | ".join(parts)

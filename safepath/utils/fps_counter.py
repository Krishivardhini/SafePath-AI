import time
from collections import deque


class FPSCounter:
    """Rolling-average FPS over the last `window` frames."""

    def __init__(self, window: int = 20):
        self._timestamps = deque(maxlen=window)

    def tick(self) -> float:
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

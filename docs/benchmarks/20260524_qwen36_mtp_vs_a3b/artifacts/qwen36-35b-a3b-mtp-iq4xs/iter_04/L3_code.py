import heapq
import time
from typing import Tuple, Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0  # To maintain FIFO for same priority
        
    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # Store as (-priority, counter, name, data)
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1
        
    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self._queue:
            return None
        _, _, name, data = heapq.heappop(self._queue)
        return name, data
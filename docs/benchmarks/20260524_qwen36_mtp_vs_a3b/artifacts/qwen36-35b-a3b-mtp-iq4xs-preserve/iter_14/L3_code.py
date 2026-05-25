import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # Negate priority for max-heap behavior with heapq (min-heap)
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._queue:
            return None
        _, _, name, data = heapq.heappop(self._queue)
        return name, data
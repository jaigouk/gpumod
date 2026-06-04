import heapq
from typing import Dict

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: Dict, priority: int = 0):
        # Use negative priority to simulate a max-heap using heapq (min-heap).
        # self._counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self._heap, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None

        # Pop the smallest element (highest priority due to negation).
        _, _, name, data = heapq.heappop(self._heap)
        return (name, data)
import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # Higher priority means larger number, but heap is min-heap.
        # So we negate priority.
        # We use counter to maintain FIFO order for same priority.
        # Tuple comparison in Python compares element by element.
        # (-priority, counter, name, data)
        heapq.heappush(self._heap, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._heap:
            return None
        # Pop smallest tuple
        _, _, name, data = heapq.heappop(self._heap)
        return (name, data)
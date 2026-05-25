import heapq
import threading # optional, but usually queues might need thread safety? The prompt doesn't specify concurrency, just "Implement priority-based job scheduling". A simple class should suffice.

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, params: dict, priority: int = 0):
        # Store negative priority for max-heap behavior using min-heap
        # Store counter for FIFO stability
        heapq.heappush(self._heap, (-priority, self._counter, name, params))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, name, params = heapq.heappop(self._heap)
        return (name, params)
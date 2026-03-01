import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        # Negate priority to use min-heap as max-heap (higher priority number = higher priority)
        # Use counter to maintain FIFO order for same priority
        entry = (-priority, self._counter, job_id, data)
        heapq.heappush(self._heap, entry)
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job_id, data = heapq.heappop(self._heap)
        return (job_id, data)
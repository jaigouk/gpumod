import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, job_id: str, job_data: dict, priority: int = 0) -> None:
        # Negate priority so that higher priority comes first in min-heap
        heapq.heappush(self._queue, (-priority, self._counter, job_id, job_data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        # Pop the item with highest priority (smallest -priority) and lowest counter
        _, _, job_id, job_data = heapq.heappop(self._queue)
        return job_id, job_data
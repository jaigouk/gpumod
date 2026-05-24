import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, job_id: str, metadata: dict, priority: int = 0):
        # Use negative priority because heapq is a min-heap
        # and higher priority numbers should be processed first.
        # Counter ensures FIFO order for same priority.
        heapq.heappush(self._queue, (-priority, self._counter, job_id, metadata))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        # Pop the item with the smallest value (most negative priority = highest priority)
        _, _, job_id, metadata = heapq.heappop(self._queue)
        return (job_id, metadata)
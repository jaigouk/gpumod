import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0  # To maintain FIFO order for same priorities

    def add_job(self, job_id: str, job_data: dict, priority: int = 0) -> None:
        # Negate priority because heapq is a min-heap, but we want higher priority first
        # Counter ensures FIFO for same priority
        heapq.heappush(self._queue, (-priority, self._counter, job_id, job_data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        # Pop the item with the highest priority (lowest negative priority value)
        # and lowest counter (earliest insertion)
        _, _, job_id, job_data = heapq.heappop(self._queue)
        return (job_id, job_data)
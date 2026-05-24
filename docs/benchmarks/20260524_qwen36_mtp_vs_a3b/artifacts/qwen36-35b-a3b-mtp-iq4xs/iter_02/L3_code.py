import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0):
        # Higher priority first, so negate priority. Counter ensures FIFO for same priority.
        heapq.heappush(self._queue, (-priority, self._counter, job_id, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        _, _, job_id, data = heapq.heappop(self._queue)
        return (job_id, data)
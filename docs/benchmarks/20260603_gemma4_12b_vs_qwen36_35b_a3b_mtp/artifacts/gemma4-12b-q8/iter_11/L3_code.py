import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0):
        # heapq is a min-heap. To make higher numbers higher priority,
        # we store priority as a negative value.
        # self._counter ensures FIFO order for identical priorities.
        heapq.heappush(self._queue, (-priority, self._counter, job_id, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        
        _, _, job_id, data = heapq.heappop(self._queue)
        return (job_id, data)
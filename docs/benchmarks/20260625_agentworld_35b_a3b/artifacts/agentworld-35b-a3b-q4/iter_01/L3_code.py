import heapq
import itertools

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = itertools.count()
    
    def add_job(self, job_id: str, job_data: dict, priority: int = 0):
        neg_priority = -priority
        count = next(self._counter)
        heapq.heappush(self._queue, (neg_priority, count, job_id, job_data))
    
    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        neg_priority, count, job_id, job_data = heapq.heappop(self._queue)
        return (job_id, job_data)
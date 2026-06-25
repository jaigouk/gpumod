import heapq
import itertools

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def add_job(self, job_id: str, params: dict, priority: int = 0):
        heapq.heappush(self._heap, (-priority, next(self._counter), job_id, params))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job_id, params = heapq.heappop(self._heap)
        return (job_id, params)
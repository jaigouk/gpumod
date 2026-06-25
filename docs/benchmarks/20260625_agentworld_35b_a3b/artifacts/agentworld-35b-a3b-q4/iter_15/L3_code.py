import heapq
import itertools

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def add_job(self, job_id: str, job_data: dict, priority: int = 0):
        count = next(self._counter)
        heapq.heappush(self._heap, (-priority, count, job_id, job_data))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job_id, job_data = heapq.heappop(self._heap)
        return (job_id, job_data)
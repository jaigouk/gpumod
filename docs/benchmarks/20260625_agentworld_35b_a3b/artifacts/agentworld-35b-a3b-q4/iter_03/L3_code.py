import heapq
import itertools

class JobQueue:
    def __init__(self):
        self.heap = []
        self.counter = itertools.count()

    def add_job(self, job_id: str, params: dict, priority: int = 0):
        seq = next(self.counter)
        heapq.heappush(self.heap, (-priority, seq, job_id, params))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.heap:
            return None
        neg_priority, seq, job_id, params = heapq.heappop(self.heap)
        return (job_id, params)
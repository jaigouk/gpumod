import heapq
import itertools

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def add_job(self, job_id: str, payload: dict, priority: int = 0):
        # Use negative priority because heapq is a min-heap.
        # Higher priority number means smaller negative value, so it comes first.
        # Counter ensures FIFO order for jobs with the same priority.
        entry = (-priority, next(self._counter), job_id, payload)
        heapq.heappush(self._heap, entry)

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        # Pop the entry with the highest priority (smallest negative priority)
        _, _, job_id, payload = heapq.heappop(self._heap)
        return (job_id, payload)
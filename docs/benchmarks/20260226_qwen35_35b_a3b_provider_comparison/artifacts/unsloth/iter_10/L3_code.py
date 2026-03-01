import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, job_id: str, job_data: dict, priority: int = 0) -> None:
        # Use negative priority because heapq is a min-heap
        # Higher priority (e.g., 2) becomes -2, which is smaller than -1 or 0
        # Counter ensures FIFO order for jobs with the same priority
        heapq.heappush(self._heap, (-priority, self._counter, job_id, job_data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job_id, job_data = heapq.heappop(self._heap)
        return (job_id, job_data)
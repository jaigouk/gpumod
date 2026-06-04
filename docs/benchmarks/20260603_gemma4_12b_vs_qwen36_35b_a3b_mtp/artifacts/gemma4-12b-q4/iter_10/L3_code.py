import heapq

    class JobQueue:
        def __init__(self):
            self._heap = []
             self._counter = 0

        def add_job(self, job_id: str, details: dict, priority: int = 0):
            # Use negative priority because heapq is a min-heap
            # counter ensures FIFO for same priority
            heapq.heappush(self._heap, (-priority, self._counter), job_id, details))
            self._counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._heap:
                return None
            _, _, job_id, details = heapq.heappop(self._heap)
            return (job_id, details)
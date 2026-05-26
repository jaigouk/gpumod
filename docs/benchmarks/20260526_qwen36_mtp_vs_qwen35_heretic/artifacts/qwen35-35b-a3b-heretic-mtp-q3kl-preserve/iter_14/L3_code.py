import heapq

    class JobQueue:
        def __init__(self):
            self._heap = []
            self._counter = 0

        def add_job(self, job_name: str, job_data: dict, priority: int = 0) -> None:
            # Negate priority because heapq is a min-heap
            # Counter ensures FIFO for same priority
            heapq.heappush(self._heap, (-priority, self._counter, job_name, job_data))
            self._counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._heap:
                return None
            _, _, job_name, job_data = heapq.heappop(self._heap)
            return (job_name, job_data)
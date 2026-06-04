import heapq

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.counter = 0

        def add_job(self, job_id: str, data: dict, priority: int = 0):
            # Heapq is a min-heap. To make higher priority come first,
            # we negate the priority. Counter ensures FIFO for same priority.
            heapq.heappush(self.heap, (-priority, self.counter, job_id, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.heap:
                return None
            _, _, job_id, data = heapq.heappop(self.heap)
            return (job_id, data)
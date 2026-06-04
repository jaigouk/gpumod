import heapq

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.counter = 0

        def add_job(self, job_id: str, data: dict, priority: int = 0):
            # Use -priority because heapq is a min-heap
            # Use self.counter to maintain FIFO for same priority
            heapq.heappush(self.heap, (-priority, self.counter, job_id, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.heap:
                return None
            priority, counter, job_id, data = heapq.heappop(self.heap)
            return (job_id, data)
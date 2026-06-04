import heapq

    class JobQueue:
        def __init__(self):
            self.jobs = []
            self.counter = 0

        def add_job(self, job_id: str, data: dict, priority: int = 0):
            # Use negative priority for max-heap behavior with heapq (min-heap)
            # Use counter to ensure FIFO for equal priorities
            heapq.heappush(self.jobs, (-priority, self.counter, job_id, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.jobs:
                return None
            priority_neg, count, job_id, data = heapq.heappop(self.jobs)
            return (job_id, data)
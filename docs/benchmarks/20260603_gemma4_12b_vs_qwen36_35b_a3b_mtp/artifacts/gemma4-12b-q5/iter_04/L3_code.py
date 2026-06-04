import heapq

class JobQueue:
    def __init__(self):
        self.jobs = []
        self.counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0):
        # Use negative priority because heapq is a min-heap (lower value = higher priority)
        # Use a counter to maintain FIFO order for jobs with the same priority
        heapq.heappush(self.jobs, (-priority, self.counter, job_id, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None

        _, _, job_id, data = heapq.heappop(self.jobs)
        return (job_id, data)
import heapq

class JobQueue:
    def __init__(self):
        self.queue = []
        self.counter = 0

    def add_job(self, job_name: str, data: dict, priority: int = 0):
        # Use negative priority to simulate a max-heap using heapq (min-heap)
        # counter ensures FIFO order for jobs with the same priority
        heapq.heappush(self.queue, (-priority, self.counter, job_name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.queue:
            return None

        _, _, job_name, data = heapq.heappop(self.queue)
        return (job_name, data)
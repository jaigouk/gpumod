import heapq

class JobQueue:
    def __init__(self):
        self.queue = []
        self.counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0):
        # heapq is a min-heap. To make higher priority values 
        # come out first, we store priority as a negative number.
        # The counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self.queue, (-priority, self.counter, job_id, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.queue:
            return None

        _, _, job_id, data = heapq.heappop(self.queue)
        return (job_id, data)
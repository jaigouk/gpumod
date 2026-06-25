import heapq

class JobQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def add_job(self, job_id: str, job_data: dict, priority: int = 0):
        entry = [-priority, self.counter, job_id, job_data]
        self.counter += 1
        heapq.heappush(self.heap, entry)

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.heap:
            return None
        _, _, job_id, job_data = heapq.heappop(self.heap)
        return (job_id, job_data)
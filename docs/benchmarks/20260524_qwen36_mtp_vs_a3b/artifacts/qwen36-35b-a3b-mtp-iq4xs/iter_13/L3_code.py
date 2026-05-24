import heapq
import time

class JobQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0  # To maintain FIFO order
        
    def add_job(self, job_id: str, job_data: dict, priority: int = 0):
        # Higher priority first -> store negative priority for min-heap
        # Lower counter first -> maintains FIFO
        heapq.heappush(self.heap, (-priority, self.counter, job_id, job_data))
        self.counter += 1
        
    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.heap:
            return None
        _, _, job_id, job_data = heapq.heappop(self.heap)
        return (job_id, job_data)
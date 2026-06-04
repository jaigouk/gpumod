import heapq
from typing import Tuple, Dict, Optional

class JobQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0):
        # Use negative priority to turn heapq (min-heap) into a max-priority queue
        # Use self.counter to ensure FIFO order for jobs with the same priority
        heapq.heappush(self.heap, (-priority, self.counter, job_id, data))
        self.counter += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]] :
        if not self.heap:
            return None

        priority_neg, count, job_id, data = heapq.heappop(self.heap)
        return (job_id, data)
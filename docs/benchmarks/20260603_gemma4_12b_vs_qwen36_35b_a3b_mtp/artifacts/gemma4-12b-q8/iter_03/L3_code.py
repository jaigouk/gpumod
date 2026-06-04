import heapq
from typing import tuple, dict, Optional

class JobQueue:
    def __init__(self):
        self.jobs = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Use negative priority because heapq is a min-heap (smallest value first).
        # Counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self.jobs, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> Optional[tuple[str, dict]]:
        if not self.jobs:
            return None

        _, _, name, data = heapq.heappop(self.jobs)
        return (name, data)
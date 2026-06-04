import heapq
    from typing import Tuple, Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = [] # List of tuples for heapq
            self.counter = 0

        def add_job(self, job_name: str, job_data: dict, priority: int = 0):
            # Higher priority = larger number.
            # heapq is a min-heap. To make it a max-priority queue,
            # use negative priority.
            heapq.heappush(self.queue, (-priority, self.counter, job_name, job_data))
            self.counter += 1

        def get_next_job(self) -> Optional[Tuple[str, Dict]]:
            if not self.queue:
                return None
            _, _, job_name, job_data = heapq.heappop(self.queue)
            return (job_name, job_data)
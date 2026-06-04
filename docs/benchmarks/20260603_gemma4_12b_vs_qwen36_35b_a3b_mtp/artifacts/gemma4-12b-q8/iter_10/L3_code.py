import heapq
    from typing import Dict, Tuple, Optional

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = 0

        def add_job(self, job_name: str, data: Dict, priority: int = 0):
            # Since heapq is a min-heap, use -priority to make higher numbers come out first
            # Use counter to maintain FIFO order for equal priorities
            heapq.heappush(self.queue, (-priority, self.counter, job_name, data))
            self.counter += 1

        def get_next_job(self) -> Optional[Tuple[str, Dict]]:
            if not self.queue:
                return None
            priority, count, job_name, data = heapq.heappop(self.queue)
            return (job_name, data)
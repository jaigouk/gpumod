import heapq
    from typing import Tuple, Dict, Optional

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.counter = 0

        def add_job(self, job_id: str, data: Dict, priority: int = 0):
            # We use -priority because heapq is a min-heap,
            # and we want the highest priority value first.
            # counter ensures FIFO for same priority.
            heapq.heappush(self.heap, (-priority, self.counter, job_id, data))
            self.counter += 1

        def get_next_job(self) -> Optional[Tuple[str, Dict]]:
            if not self.heap:
                return None
            priority_neg, count, job_id, data = heapq.heappop(self.heap)
            return (job_id, data)
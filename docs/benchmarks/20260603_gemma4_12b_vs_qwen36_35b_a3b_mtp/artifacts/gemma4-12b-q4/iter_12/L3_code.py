import heapq
    from typing import Tuple, Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = 0

        def add_job(self, job_id: str, data: Dict, priority: int = 0):
            # We use negative priority because heapq is a min-heap,
            # but we want higher priority numbers to be processed first.
            # We use a counter to ensure FIFO for same priority.
            heapq.heappush(self.queue, (-priority, self.counter, job_id, data))
            self.counter += 1

        def get_next_job(self) -> Optional[Tuple[str, Dict]] :
            if not self.queue:
                return None
            priority_neg, count, job_id, data = heapq.heappop(self.queue)
            return (job_id, data)
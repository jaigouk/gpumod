import heapq
    from typing import Tuple, Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.count = 0

        def add_job(self, job_name: str, data: dict, priority: int = 0):
            # Use negative priority for max-heap behavior
            # Use self.count to maintain FIFO for equal priorities
            heapq.heappush(self.queue, (-priority, self.count, job_name, data))
            self.count += 1

        def get_next_job(self) -> Optional[Tuple[str, dict]]:
            if not self.queue:
                return None
            _, _, job_name, data = heapq.heappop(self.queue)
            return (job_name, data)
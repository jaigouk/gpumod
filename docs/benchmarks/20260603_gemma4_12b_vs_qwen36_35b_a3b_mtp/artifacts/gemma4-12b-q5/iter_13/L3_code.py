import heapq
    from typing import Tuple, Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = 0

        def add_job(self, name: str, data: Dict, priority: int = 0):
            # heapq is a min-heap. To act as a max-priority queue:
            # We use -priority.
            # We use self.counter to ensure FIFO for same priorities.
            heapq.heappush(self.queue, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> Optional[Tuple[str, Dict]]:
            if not self.queue:
                return None
            priority, count, name, data = heapq.heappop(self.queue)
            return (name, data)
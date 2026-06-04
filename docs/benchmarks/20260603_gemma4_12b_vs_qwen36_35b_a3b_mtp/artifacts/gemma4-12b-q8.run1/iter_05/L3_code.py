import heapq
    from typing import Tuple, Dict, Optional

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Use -priority because heapq is a min-heap
            # counter ensures FIFO for same priority
            heapq.heappush(self.heap, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> Optional[Tuple[str, dict]]:
            if not self.heap:
                return None
            _, _, name, data = heapq.heappop(self.heap)
            return (name, data)
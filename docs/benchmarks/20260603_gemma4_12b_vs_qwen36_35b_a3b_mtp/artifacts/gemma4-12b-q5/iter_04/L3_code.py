import heapq
    from typing import Tuple, Dict, Optional

    class JobQueue:
        def __init__(self):
            self._heap = []
            self._counter = 0

        def add_job(self, job_name: str, data: Dict, priority: int = 0):
            # Use negative priority because heapq is a min-heap
            # Counter ensures FIFO for equal priorities
            heapq.heappush(self._heap, (-priority, self._counter, job_name, data))
            self._counter += 1

        def get_next_job(self) -> Optional[Tuple[str, Dict]]:
            if not self._heap:
                return None
            _, _, job_name, data = heapq.heappop(self._heap)
            return (job_name, data)
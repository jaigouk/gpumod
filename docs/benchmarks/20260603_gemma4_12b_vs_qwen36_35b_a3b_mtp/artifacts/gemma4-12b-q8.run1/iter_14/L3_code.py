import heapq
    from typing import Tuple, Dict, Optional

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._counter = 0

        def add_job(self, name: str, data: Dict, priority: int = 0):
            # Use negative priority to make it a max-heap (higher number first)
            # Counter ensures FIFO for same priorities
            heapq.heappush(self._queue, (-priority, self._counter, name, data))
            self._counter += 1

        def get_next_job(self) -> Optional[Tuple[str, Dict]]:
            if not self._queue:
                return None
            _, _, name, data = heapq.heappop(self._queue)
            return (name, data)
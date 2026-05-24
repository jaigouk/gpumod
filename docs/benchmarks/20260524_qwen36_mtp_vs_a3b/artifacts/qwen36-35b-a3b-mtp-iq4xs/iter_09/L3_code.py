import heapq
from typing import Tuple, Dict, Optional

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, metadata: dict, priority: int = 0) -> None:
        # Use negative priority for max-heap behavior with min-heap
        heapq.heappush(self._heap, (-priority, self._counter, name, metadata))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._heap:
            return None
        _, _, name, metadata = heapq.heappop(self._heap)
        return name, metadata
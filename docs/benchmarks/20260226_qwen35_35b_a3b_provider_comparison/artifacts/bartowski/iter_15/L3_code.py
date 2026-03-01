import heapq
from typing import Dict, Optional, Tuple

class JobQueue:
    def __init__(self):
        self._heap = []
        self._sequence = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # Negate priority so higher numbers (2, 1) are popped before lower (0)
        # Sequence ensures FIFO order for same priority
        heapq.heappush(self._heap, (-priority, self._sequence, name, data))
        self._sequence += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._heap:
            return None
        _, _, name, data = heapq.heappop(self._heap)
        return (name, data)
import heapq
from itertools import count

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = count()

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Using -priority because heapq is a min-heap, 
        # but we want higher numbers to be processed first.
        # The counter ensures FIFO order for items with the same priority.
        count_val = next(self._counter)
        heapq.heappush(self._queue, (-priority, count_val, name, data))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        _, _, name, data = heapq.heappop(self._queue)
        return name, data
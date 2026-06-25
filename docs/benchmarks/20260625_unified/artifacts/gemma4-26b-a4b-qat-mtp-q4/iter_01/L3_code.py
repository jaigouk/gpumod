import heapq
from itertools import count

class JobQueue:
    def __init__(self) -> None:
        self._queue = []
        self._counter = count()

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # We negate priority to use heapq (a min-heap) as a max-priority queue.
        # We use a counter to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, next(self._counter), name, data))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        # Pop the job with the lowest value in the heap.
        # Because priority is negated, the smallest value is the highest priority.
        _, _, name, data = heapq.heappop(self._queue)
        return name, data
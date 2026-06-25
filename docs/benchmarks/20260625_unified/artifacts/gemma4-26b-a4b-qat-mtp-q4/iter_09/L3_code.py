import heapq
from itertools import count

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = count()

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # heapq is a min-heap. To process higher priority numbers first, 
        # we store the priority as a negative value.
        # We include a monotonic counter to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, next(self._counter), name, data))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        _, _, name, data = heapq.heappop(self._queue)
        return name, data
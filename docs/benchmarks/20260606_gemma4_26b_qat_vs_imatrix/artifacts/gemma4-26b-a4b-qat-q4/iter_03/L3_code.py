import heapq
from itertools import count

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = count()

    def add_job(self, name: str, metadata: dict, priority: int = 0):
        # Use -priority to turn heapq's min-heap into a max-priority queue
        # Use next(self._counter) to ensure FIFO order for jobs with the same priority
        entry = (-priority, next(self._counter), name, metadata)
        heapq.heappush(self._queue, entry)

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        _, _, name, metadata = heapq.heappop(self._queue)
        return name, metadata
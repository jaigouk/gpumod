import heapq
from itertools import count

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = count()

    def add_job(self, name: str, metadata: dict, priority: int = 0) -> None:
        """
        Adds a job to the queue.
        Higher priority values are processed first.
        Same priority jobs follow FIFO order.
        """
        # heapq is a min-heap, so we negate the priority to make higher 
        # numbers "smaller" for the heap algorithm. 
        # The counter ensures FIFO for identical priorities.
        heapq.heappush(self._queue, (-priority, next(self._counter), name, metadata))

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job as (name, metadata).
        Returns None if the queue is empty.
        """
        if not self._queue:
            return None

        _, _, name, metadata = heapq.heappop(self._queue)
        return name, metadata
import heapq
import itertools

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = itertools.count()

    def add_job(self, name: str, metadata: dict, priority: int = 0):
        """
        Adds a job to the queue.
        Higher priority values are processed first.
        Jobs with same priority follow FIFO.
        """
        # We use -priority because heapq is a min-heap, 
        # but we want higher priority numbers to come out first.
        # We use self._counter to ensure FIFO for identical priorities.
        count = next(self._counter)
        heapq.heappush(self._queue, (-priority, count, name, metadata))

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job as (name, metadata).
        Returns None if the queue is empty.
        """
        if not self._queue:
            return None

        _, _, name, metadata = heapq.heappop(self._queue)
        return (name, metadata)
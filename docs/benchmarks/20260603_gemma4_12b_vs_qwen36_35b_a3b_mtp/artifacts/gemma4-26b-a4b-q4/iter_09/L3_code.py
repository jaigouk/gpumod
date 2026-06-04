import heapq
import itertools

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def add_job(self, name: str, data: dict, priority: int = 0):
        """
        Adds a job to the queue.
        Higher priority values are processed first.
        Same priority jobs follow FIFO order.
        """
        count = next(self._counter)
        # Use -priority because heapq is a min-heap, but we want highest priority first.
        # count ensures FIFO order for identical priorities.
        heapq.heappush(self._heap, (-priority, count, name, data))

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves and removes the highest priority job.
        Returns (name, data) or None if the queue is empty.
        """
        if not self._heap:
            return None
        
        _, _, name, data = heapq.heappop(self._heap)
        return name, data
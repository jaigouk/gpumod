import heapq
import itertools

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        """
        Adds a job to the queue.
        Priority: higher values are processed first.
        Tie-breaking: FIFO order for same priority.
        """
        # We use -priority because heapq is a min-heap, 
        # and we want the highest priority number to be popped first.
        # The counter ensures FIFO order for items with the same priority.
        count = next(self._counter)
        heapq.heappush(self._heap, (-priority, count, name, data))

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves and removes the highest priority job from the queue.
        Returns None if the queue is empty.
        """
        if not self._heap:
            return None
        
        _, _, name, data = heapq.heappop(self._heap)
        return name, data
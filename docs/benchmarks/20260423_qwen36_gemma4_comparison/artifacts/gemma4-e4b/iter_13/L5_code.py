# queue/priority.py
import heapq
from typing import List, Tuple, Any

class PriorityQueue:
    """
    A min-heap based priority queue.
    Items are stored as (priority, insertion_index, job_data).
    """
    def __init__(self):
        # The heap stores tuples: (priority, index, job)
        self._queue: List[Tuple[int, int, Any]] = []
        self._index = 0

    def put(self, priority: int, job: Any):
        """Adds a job to the queue with a given priority (lower number = higher priority)."""
        heapq.heappush(self._queue, (priority, self._index, job))
        self._index += 1

    def get(self) -> Any:
        """Removes and returns the job with the highest priority."""
        if not self._queue:
            raise IndexError("get from an empty priority queue")
        # Pop the smallest item (highest priority)
        priority, _, job = heapq.heappop(self._queue)
        return job

    def empty(self) -> bool:
        """Checks if the queue is empty."""
        return not self._queue

    def qsize(self) -> int:
        """Returns the number of items in the queue."""
        return len(self._queue)
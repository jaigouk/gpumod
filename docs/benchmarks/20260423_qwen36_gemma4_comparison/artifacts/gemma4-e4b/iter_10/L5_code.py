import heapq
from typing import Any, List, Tuple

class PriorityQueue:
    """
    A min-heap implementation used to store jobs based on their priority.
    (Lower priority value means higher urgency/priority).
    """
    def __init__(self):
        # Heap stores tuples: (priority, timestamp, item)
        # Timestamp is used to ensure stable ordering for jobs with the same priority
        self._queue: List[Tuple[int, float, Any]] = []
        self._counter = 0  # Tie-breaker counter

    def put(self, priority: int, item: Any):
        """Adds an item to the queue."""
        # Use current time and a counter for deterministic ordering
        timestamp = heapq.nlargest(1, [float(i) for i in self._queue])[0] if self._queue else 0.0
        
        heapq.heappush(self._queue, (priority, self._counter, item))
        self._counter += 1

    def get(self) -> Any:
        """Removes and returns the highest priority item."""
        if not self._queue:
            raise IndexError("get from empty priority queue")
        
        # The item is the third element in the tuple
        priority, _, item = heapq.heappop(self._queue)
        return item

    def empty(self) -> bool:
        """Checks if the queue is empty."""
        return not self._queue

    def qsize(self) -> int:
        """Returns the number of items in the queue."""
        return len(self._queue)
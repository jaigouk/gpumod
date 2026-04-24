import heapq
from typing import List, Tuple, Any

class PriorityQueue:
    """
    Implements a min-heap based priority queue.
    Items are stored as (priority, insertion_order, item).
    """
    def __init__(self):
        # The heap stores tuples: (priority, count, item)
        self._queue: List[Tuple[int, int, Any]] = []
        self._counter = 0  # Used to ensure FIFO order for items with the same priority

    def put(self, item: Any, priority: int):
        """Adds an item to the queue with a given priority."""
        # Lower priority number means higher priority (standard min-heap behavior)
        entry = (priority, self._counter, item)
        heapq.heappush(self._queue, entry)
        self._counter += 1

    def get(self) -> Any:
        """Removes and returns the highest priority item."""
        if not self.empty():
            # Pop the root item (which has the lowest priority number)
            priority, count, item = heapq.heappop(self._queue)
            return item
        raise IndexError("get from empty priority queue")

    def empty(self) -> bool:
        """Checks if the queue is empty."""
        return not self._queue

    def qsize(self) -> int:
        """Returns the number of items in the queue."""
        return len(self._queue)
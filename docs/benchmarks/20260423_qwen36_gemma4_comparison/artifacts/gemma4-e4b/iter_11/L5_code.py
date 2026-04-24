import heapq
from typing import List, Tuple, Any

class PriorityQueue:
    """
    A min-heap implementation of a Priority Queue.
    Items are ordered based on their priority (lower number = higher priority).
    """
    def __init__(self):
        # The heap stores tuples: (priority, entry_count, item)
        # entry_count is used to ensure stable sorting (FIFO for same priority)
        self._heap: List[Tuple[int, int, Any]] = []
        self._counter = 0

    def put(self, item: Any, priority: int):
        """Adds an item to the queue with a given priority."""
        entry = (priority, self._counter, item)
        heapq.heappush(self._heap, entry)
        self._counter += 1

    def get(self) -> Any:
        """Removes and returns the item with the highest priority."""
        if not self._heap:
            raise IndexError("get from empty priority queue")
        # We pop the tuple, but only return the actual item (index 2)
        return heapq.heappop(self._heap)[2]

    def empty(self) -> bool:
        """Checks if the queue is empty."""
        return not self._heap

    def qsize(self) -> int:
        """Returns the number of items in the queue."""
        return len(self._heap)
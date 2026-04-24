import heapq
from typing import Any, List, Tuple

class PriorityQueue:
    """
    A min-heap implementation of a Priority Queue.
    Items are stored as (priority, insertion_index, item).
    """
    def __init__(self):
        # The heap stores tuples: (priority, counter, item)
        self._heap: List[Tuple[int, int, Any]] = []
        self._counter = 0  # Tie-breaker for stable ordering

    def push(self, item: Any, priority: int):
        """Adds an item to the queue with a specified priority."""
        # Lower priority number means higher urgency
        entry = (priority, self._counter, item)
        heapq.heappush(self._heap, entry)
        self._counter += 1

    def pop(self) -> Any:
        """Removes and returns the item with the highest priority (lowest priority number)."""
        if not self._heap:
            raise IndexError("pop from empty priority queue")
        # We only return the actual item (index 2)
        return heapq.heappop(self._heap)[2]

    def empty(self) -> bool:
        """Checks if the queue is empty."""
        return not self._heap

    def qsize(self) -> int:
        """Returns the number of items in the queue."""
        return len(self._heap)
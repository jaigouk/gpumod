import heapq
from typing import Any, Optional, Tuple

class PriorityQueue:
    """
    A thread-safe priority queue implementation using heapq.
    Stores items as tuples of (priority, timestamp, item).
    """

    def __init__(self):
        self._heap: list = []
        self._counter = 0  # To ensure stable sorting for items with same priority

    def push(self, priority: int, item: Any) -> None:
        """Push an item with a specific priority onto the queue."""
        entry = (priority, self._counter, item)
        self._counter += 1
        heapq.heappush(self._heap, entry)

    def pop(self) -> Tuple[int, Any]:
        """Pop the item with the highest priority (lowest value)."""
        if not self._heap:
            raise IndexError("Priority queue is empty")
        priority, _, item = heapq.heappop(self._heap)
        return priority, item

    def peek(self) -> Optional[Tuple[int, Any]]:
        """Return the highest priority item without removing it."""
        if not self._heap:
            return None
        priority, _, item = self._heap[0]
        return priority, item

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0
import heapq
from typing import List, Any, Tuple

# We assume Job is defined elsewhere (in core.py) and is comparable.
# For robustness, we'll assume the Job object has a 'priority' attribute.

class PriorityQueue:
    """
    A min-heap implementation of a Priority Queue.
    Items are stored as (priority, insertion_index, item).
    """
    def __init__(self):
        # Heap stores tuples: (priority, index, item)
        self._heap = []
        self._index = 0

    def push(self, item: Any, priority: int):
        """Adds an item with a given priority to the queue."""
        # We use the index as a tie-breaker to ensure stable ordering
        entry = (priority, self._index, item)
        heapq.heappush(self._heap, entry)
        self._index += 1

    def pop(self) -> Any:
        """Removes and returns the item with the highest priority (lowest number)."""
        if not self._heap:
            raise IndexError("pop from empty PriorityQueue")
        # The item is the third element in the tuple
        _, _, item = heapq.heappop(self._heap)
        return item

    def is_empty(self) -> bool:
        """Checks if the queue is empty."""
        return len(self._heap) == 0

    def __len__(self):
        return len(self._heap)
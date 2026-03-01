import heapq
from typing import Any, Optional
from dataclasses import dataclass, field

@dataclass(order=True)
class PriorityItem:
    """Helper class to handle items in the heap to ensure stable sorting."""
    priority: int
    sequence: int
    item: Any = field(compare=False)

class PriorityQueue:
    """A thread-safe priority queue implementation using heapq."""
    
    def __init__(self):
        self._heap: list[PriorityItem] = []
        self._counter: int = 0
        self._lock = None # Placeholder for threading.Lock if needed

    def push(self, item: Any, priority: int) -> None:
        """Push an item with a specific priority."""
        if self._lock:
            self._lock.acquire()
        try:
            item = PriorityItem(priority=priority, sequence=self._counter, item=item)
            heapq.heappush(self._heap, item)
            self._counter += 1
        finally:
            if self._lock:
                self._lock.release()

    def pop(self) -> Optional[Any]:
        """Pop the item with the highest priority (lowest value)."""
        if self._lock:
            self._lock.acquire()
        try:
            if not self._heap:
                return None
            item = heapq.heappop(self._heap)
            return item.item
        finally:
            if self._lock:
                self._lock.release()

    def peek(self) -> Optional[Any]:
        """Return the highest priority item without removing it."""
        if self._lock:
            self._lock.acquire()
        try:
            if not self._heap:
                return None
            return self._heap[0].item
        finally:
            if self._lock:
                self._lock.release()

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._heap) == 0

    def size(self) -> int:
        """Return the number of items in the queue."""
        return len(self._heap)
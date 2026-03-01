import heapq
from typing import Any, Optional

# Import Job from core. 
# Note: This creates a circular dependency with core.py, 
# but it is safe because Job is defined before JobQueue uses PriorityQueue.
from .core import Job

class PriorityQueue:
    """
    A thread-safe priority queue implementation using a min-heap.
    Jobs with lower priority values are processed first.
    """
    
    def __init__(self):
        self._heap: list[tuple[int, int, Job]] = []
        self._counter = 0  # To maintain FIFO order for same-priority jobs

    def push(self, job: Job) -> None:
        """Add a job to the queue."""
        # Tuple: (priority, insertion_order, job)
        # insertion_order ensures stable sorting (FIFO) for same priority
        heapq.heappush(self._heap, (job.priority, self._counter, job))
        self._counter += 1

    def pop(self) -> Optional[Job]:
        """Remove and return the highest priority job."""
        if not self._heap:
            return None
        _, _, job = heapq.heappop(self._heap)
        return job

    def peek(self) -> Optional[Job]:
        """Return the highest priority job without removing it."""
        if not self._heap:
            return None
        return self._heap[0][2]

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)
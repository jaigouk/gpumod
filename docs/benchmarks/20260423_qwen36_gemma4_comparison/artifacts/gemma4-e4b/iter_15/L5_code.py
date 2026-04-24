import heapq
from typing import List, Tuple, Any

class PriorityQueue:
    """
    A min-heap implementation used to store Jobs based on their priority.
    Lower priority number means higher priority.
    """
    def __init__(self):
        # The heap stores tuples: (priority, insertion_index, job)
        # insertion_index is used to maintain stable ordering for items with the same priority.
        self._queue: List[Tuple[int, int, Any]] = []
        self._counter = 0

    def put(self, priority: int, job: Any) -> None:
        """Adds a job to the queue."""
        heapq.heappush(self._queue, (priority, self._counter, job))
        self._counter += 1

    def get(self) -> Any:
        """Removes and returns the job with the highest priority."""
        if not self._queue:
            raise IndexError("get from empty priority queue")
        # The job object is the third element in the tuple
        return heapq.heappop(self._queue)[2]

    def empty(self) -> bool:
        """Checks if the queue is empty."""
        return not self._queue
    
    def __len__(self) -> int:
        return len(self._queue)

# Example usage (for testing/internal use):
# pq = PriorityQueue()
# pq.put(5, "Low priority task")
# pq.put(1, "High priority task")
# print(pq.get()) # Output: High priority task
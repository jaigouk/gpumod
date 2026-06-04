import heapq
import itertools

class JobQueue:
    def __init__(self):
        # Using a list to act as a heap
        self._queue = []
        # A counter to ensure FIFO order for jobs with the same priority
        self._counter = itertools.count()

    def add_job(self, job_id: str, data: dict, priority: int = 0):
        """
        Adds a job to the queue with a given priority level.
        Higher priority levels are processed first.
        """
        # We use negative priority because heapq is a min-heap. 
        # This ensures that a higher priority number (e.g., 2) 
        # becomes a smaller value (-2) and is popped first.
        # The counter acts as a tie-breaker to maintain FIFO order.
        entry = (-priority, next(self._counter), job_id, data)
        heapq.heappush(self._queue, entry)

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job as a tuple (job_id, data).
        Returns None if the queue is empty.
        """
        if not self._queue:
            return None

        # Pop the smallest value from the heap (which is the highest priority)
        _, _, job_id, data = heapq.heappop(self._queue)
        return (job_id, data)
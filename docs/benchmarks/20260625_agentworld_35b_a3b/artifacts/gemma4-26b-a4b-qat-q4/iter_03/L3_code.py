import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        """
        Adds a job to the queue.
        Higher priority values are processed first. 
        Jobs with same priority follow FIFO order.
        """
        # We store priority as a negative number because heapq is a min-heap.
        # The counter ensures FIFO behavior for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job as a tuple of (name, data).
        Returns None if the queue is empty.
        """
        if not self._queue:
            return None

        # Pop the element with the smallest negative priority (highest actual priority)
        _, _, name, data = heapq.heappop(self._queue)
        return name, data
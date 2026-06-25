import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, metadata: dict, priority: int = 0):
        """
        Adds a job to the queue.
        Priority is inverted to work with heapq (min-heap).
        A counter is used to maintain FIFO order for identical priorities.
        """
        heapq.heappush(self._queue, (-priority, self._counter, name, metadata))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job as a (name, metadata) tuple.
        Returns None if the queue is empty.
        """
        if not self._queue:
            return None

        _, _, name, metadata = heapq.heappop(self._queue)
        return (name, metadata)
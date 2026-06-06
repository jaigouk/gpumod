import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        """
        Adds a job to the queue.
        Uses -priority to maintain a max-heap behavior for priority values.
        Uses self._counter to maintain FIFO order for jobs with the same priority.
        """
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job as (name, data).
        Returns None if the queue is empty.
        """
        if not self._queue:
            return None

        _, _, name, data = heapq.heappop(self._queue)
        return name, data
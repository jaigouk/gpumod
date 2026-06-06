import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        """
        Adds a job to the queue. 
        Higher priority numbers are processed first.
        Same priority jobs follow FIFO order.
        """
        # heapq is a min-heap. We use -priority to turn it into a max-priority queue.
        # We use self._counter to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job as a (name, data) tuple,
        or None if the queue is empty.
        """
        if not self._queue:
            return None

        # Pop the element with the lowest (-priority) value, then lowest counter
        _, _, name, data = heapq.heappop(self._queue)
        return name, data
import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # heapq is a min-heap. To treat higher numbers as higher priority,
        # we store the priority as a negative value.
        # We include a counter to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        priority_neg, count, name, data = heapq.heappop(self._queue)
        return (name, data)
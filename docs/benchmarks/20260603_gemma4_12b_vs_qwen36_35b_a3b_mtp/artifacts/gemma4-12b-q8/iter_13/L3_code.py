import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, metadata: dict, priority: int = 0):
        # heapq is a min-heap. To make higher numbers higher priority, 
        # we store priority as a negative number.
        # self._counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, metadata))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        priority_neg, count, name, metadata = heapq.heappop(self._queue)
        return (name, metadata)
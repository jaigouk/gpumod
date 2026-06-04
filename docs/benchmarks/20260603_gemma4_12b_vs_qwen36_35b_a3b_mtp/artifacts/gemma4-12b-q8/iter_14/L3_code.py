import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # heapq is a min-heap. To treat higher priority numbers 
        # (e.g., 2) as higher priority, we negate the value.
        # The counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        # pop the item with the lowest negated priority (highest actual priority)
        _, _, name, data = heapq.heappop(self._queue)
        return (name, data)
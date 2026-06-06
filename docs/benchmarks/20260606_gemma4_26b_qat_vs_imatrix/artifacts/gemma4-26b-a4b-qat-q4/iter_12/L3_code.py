import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Use -priority because heapq is a min-heap, and we want higher priority first.
        # Use self._index as a tie-breaker to ensure FIFO for identical priorities.
        heapq.heappush(self._queue, (-priority, self._index, name, data))
        self._index += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        _, _, name, data = heapq.heappop(self._queue)
        return name, data
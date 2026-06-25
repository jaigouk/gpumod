import heapq

class JobQueue:
    def __init__(self) -> None:
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # We use -priority because heapq is a min-heap (we want highest number first).
        # We use self._counter as a tie-breaker to ensure FIFO for same priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        # Pop the element with the smallest negated priority (highest actual priority)
        _, _, name, data = heapq.heappop(self._queue)
        return name, data
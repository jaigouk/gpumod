import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # We use -priority because heapq is a min-heap, but we want higher priority first.
        # We use self._counter to maintain FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        # heappop returns the tuple with the smallest -priority (highest priority)
        # and the smallest counter (oldest job)
        _, _, name, data = heapq.heappop(self._queue)
        return name, data
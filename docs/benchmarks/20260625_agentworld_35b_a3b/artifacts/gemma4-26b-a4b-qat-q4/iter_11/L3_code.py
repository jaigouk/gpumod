import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # We use -priority because heapq is a min-heap, and we want higher numbers first.
        # We use self._counter to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        # Pop the item with the lowest negative priority (highest actual priority)
        # and the lowest counter (oldest insertion)
        _, _, name, data = heapq.heappop(self._queue)
        return name, data
import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # heapq is a min-heap. To make higher priority values come first, 
        # we store priority as a negative number.
        # self._counter ensures FIFO order for jobs with identical priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        # Pop the item with the lowest negative priority (highest actual priority)
        _, _, name, data = heapq.heappop(self._queue)
        return name, data
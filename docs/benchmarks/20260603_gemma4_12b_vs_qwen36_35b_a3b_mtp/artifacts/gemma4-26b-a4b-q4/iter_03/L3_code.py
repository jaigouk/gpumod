import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._count = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # heapq is a min-heap. To make higher priority numbers come out first,
        # we store the priority as a negative value.
        # We include self._count to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._count, name, data))
        self._count += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        # Pop the element with the lowest value (which is the highest priority)
        _, _, name, data = heapq.heappop(self._queue)
        return (name, data)
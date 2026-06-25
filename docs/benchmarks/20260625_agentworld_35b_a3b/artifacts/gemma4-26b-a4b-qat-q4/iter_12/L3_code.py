import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # We use -priority because heapq is a min-heap, but we want high numbers first.
        # We include self._index to maintain FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._index, name, data))
        self._index += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        # Pop the element with the lowest (-priority, index)
        _, _, name, data = heapq.heappop(self._queue)
        return (name, data)
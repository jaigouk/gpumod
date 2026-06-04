import heapq
    import itertools

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._counter = itertools.count()

        def add_job(self, name: str, data: dict, priority: int = 0):
            # We use -priority because heapq is a min-heap.
            # We use self._counter to ensure FIFO for equal priorities.
            heapq.heappush(self._queue, (-priority, next(self._counter), name, data))

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._queue:
                return None
            _, _, name, data = heapq.heappop(self._queue)
            return name, data
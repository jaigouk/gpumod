import heapq
    import itertools

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._counter = itertools.count()

        def add_job(self, name: str, data: dict, priority: int = 0):
            count = next(self._counter)
            # heapq is a min-heap. To make it a max-priority heap,
            # we negate the priority. 
            # To maintain FIFO for same priority, we use the counter.
            heapq.heappush(self._queue, (-priority, count, name, data))

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._queue:
                return None
            _, _, name, data = heapq.heappop(self._queue)
            return name, data
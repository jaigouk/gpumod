import heapq
    from itertools import count

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._counter = count()

        def add_job(self, name: str, data: dict, priority: int = 0):
            # heapq is a min-heap. To make it max-priority, use -priority.
            # counter ensures FIFO for same priority.
            heapq.heappush(self._queue, (-priority, next(self._counter), name, data))

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._queue:
                return None
            _, _, name, data = heapq.heappop(self._queue)
            return (name, data)
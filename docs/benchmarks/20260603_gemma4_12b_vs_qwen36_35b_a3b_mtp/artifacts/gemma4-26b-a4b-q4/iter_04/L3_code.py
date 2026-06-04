import heapq
    from itertools import count

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._counter = count()

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Using negative priority to turn min-heap into max-priority-heap
            # Using counter to ensure FIFO for same priority
            heapq.heappush(self._queue, (-priority, next(self._counter), name, data))

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._queue:
                return None
            priority_neg, _, name, data = heapq.heappop(self._queue)
            return name, data
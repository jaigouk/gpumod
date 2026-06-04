import heapq
    from itertools import count

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._counter = count()

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Use negative priority for max-heap behavior using heapq (min-heap)
            # Use self._counter for FIFO
            heapq.heappush(self._queue, (-priority, next(self._counter), name, data))

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._queue:
                return None
            # Pop the smallest element (which is highest priority due to negation)
            _, _, name, data = heapq.heappop(self._queue)
            return name, data
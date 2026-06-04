import heapq

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Use negative priority because heapq is a min-heap
            # Use counter to maintain FIFO for same priority
            heapq.heappush(self._queue, (-priority, self._counter, name, data))
            self._counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._queue:
                return None
            priority, count, name, data = heapq.heappop(self._queue)
            return (name, data)
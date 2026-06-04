import heapq

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._index = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Use negative priority for max-heap behavior using heapq (min-heap)
            # Use self._index to ensure FIFO for same priority
            heapq.heappush(self._queue, (-priority, self._index, name, data))
            self._index += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._queue:
                return None
            _, _, name, data = heapq.heappop(self._queue)
            return name, data
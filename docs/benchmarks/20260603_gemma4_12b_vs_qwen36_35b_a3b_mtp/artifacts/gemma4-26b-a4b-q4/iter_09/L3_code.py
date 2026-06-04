import heapq

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # heapq is a min-heap.
            # To make higher priority (2) come before lower (0), we use -priority.
            # To ensure FIFO for same priority, we use self._counter.
            heapq.heappush(self._queue, (-priority, self._counter, name, data))
            self._counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._queue:
                return None
            # Pop the smallest element (highest priority/lowest counter)
            _, _, name, data = heapq.heappop(self._queue)
            return (name, data)
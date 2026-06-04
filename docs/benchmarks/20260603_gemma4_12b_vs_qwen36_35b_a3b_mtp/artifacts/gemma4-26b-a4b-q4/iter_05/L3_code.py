import heapq
    import itertools

    class JobQueue:
        def __init__(self):
            self._heap = []
            self._counter = itertools.count()

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Using -priority for max-heap behavior with heapq (min-heap)
            # Using self._counter to maintain FIFO for same priority
            count = next(self._counter)
            heapq.heappush(self._heap, (-priority, count, name, data))

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._heap:
                return None
            _, _, name, data = heapq.heappop(self._heap)
            return (name, data)
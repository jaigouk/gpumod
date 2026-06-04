import heapq

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._count = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Use -priority to turn min-heap into max-heap for priority
            # Use self._count to maintain FIFO for equal priorities
            heapq.heappush(self._queue, (-priority, self._count, name, data)))
            self._count += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._queue:
                return None
            priority_neg, count, name, data = heapq.heappop(self._queue)
            return (name, data)
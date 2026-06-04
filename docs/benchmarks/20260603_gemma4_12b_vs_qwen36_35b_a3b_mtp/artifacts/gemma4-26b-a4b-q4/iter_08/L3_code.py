import heapq
    from itertools import count

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = count()

        def add_job(self, name: str, metadata: dict, priority: int = 0):
            # heapq is a min-heap.
            # Higher priority (e.g., 2) should come first.
            # So we use -priority.
            # To handle FIFO for same priority, we use an incrementing counter.
            count = next(self.counter)
            heapq.heappush(self.queue, (-priority, count, name, metadata))

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.queue:
                return None
            _, _, name, metadata = heapq.heappop(self.queue)
            return (name, metadata)
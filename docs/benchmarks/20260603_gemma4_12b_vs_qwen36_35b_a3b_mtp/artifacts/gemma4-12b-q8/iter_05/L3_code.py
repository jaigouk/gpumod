import heapq
    import itertools

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = itertools.count()

        def add_job(self, name: str, data: dict, priority: int = 0):
            # heapq is a min-heap. 
            # To make higher priority (e.g., 2) come out before lower (e.g., 0),
            # we store priority as a negative number.
            # To maintain FIFO for same priority, we use a counter.
            heapq.heappush(self.queue, (-priority, next(self.counter), name, data))

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.queue:
                return None
            _, _, name, data = heapq.heappop(self.queue)
            return name, data
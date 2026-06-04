import heapq

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Use negative priority for max-heap behavior with heapq (min-heap)
            # Use counter to maintain FIFO for same priority
            heapq.heappush(self.queue, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.queue:
                return None
            priority_neg, count, name, data = heapq.heappop(self.queue)
            return (name, data)
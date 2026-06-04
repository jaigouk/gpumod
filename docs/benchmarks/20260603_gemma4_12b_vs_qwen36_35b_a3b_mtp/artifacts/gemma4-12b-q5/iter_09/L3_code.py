import heapq

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Using negative priority because heapq is a min-heap
            # and we want higher numbers to be popped first.
            # counter ensures FIFO for same priorities.
            heapq.heappush(self.queue, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.queue:
                return None
            _, _, name, data = heapq.heappop(self.queue)
            return (name, data)
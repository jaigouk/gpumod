import heapq

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Negate priority because heapq is a min-heap,
            # but we want higher priority numbers to be processed first.
            heapq.heappush(self.queue, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.queue:
                return None
            _, _, name, data = heapq.heappop(self.queue)
            return (name, data)
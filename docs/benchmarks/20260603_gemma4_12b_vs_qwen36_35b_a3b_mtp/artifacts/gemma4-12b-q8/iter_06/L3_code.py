import heapq

class JobQueue:
    def __init__(self):
        self.queue = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # heapq is a min-heap. To treat higher numbers as higher priority,
        # we store the priority as a negative number.
        # We use a counter to ensure FIFO behavior for equal priorities.
        heapq.heappush(self.queue, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.queue:
            return None

        _, _, name, data = heapq.heappop(self.queue)
        return (name, data)
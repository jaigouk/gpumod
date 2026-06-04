import heapq

class JobQueue:
    def __init__(self):
        self.queue = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Since heapq is a min-heap, we store priority as a negative number
        # to ensure higher priority values are popped first.
        # The counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self.queue, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.queue:
            return None

        _, _, name, data = heapq.heappop(self.queue)
        return (name, data)
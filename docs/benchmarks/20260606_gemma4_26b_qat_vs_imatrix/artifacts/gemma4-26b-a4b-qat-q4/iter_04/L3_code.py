import heapq

class JobQueue:
    def __init__(self):
        self.queue = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Use -priority for max-heap behavior (higher number = higher priority)
        # Use self.counter to maintain FIFO for jobs with the same priority
        heapq.heappush(self.queue, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.queue:
            return None

        # Pop the element with the lowest negative priority (highest actual priority)
        _, _, name, data = heapq.heappop(self.queue)
        return (name, data)
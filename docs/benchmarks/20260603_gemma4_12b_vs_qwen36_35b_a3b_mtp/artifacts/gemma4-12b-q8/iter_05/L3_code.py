import heapq

class JobQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Negate priority to turn min-heap into max-heap
        # Use counter to maintain FIFO order for identical priorities
        heapq.heappush(self.heap, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.heap:
            return None

        _, _, name, data = heapq.heappop(self.heap)
        return (name, data)
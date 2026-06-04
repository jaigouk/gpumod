import heapq

class JobQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Using negative priority because heapq is a min-heap.
        # Higher priority numbers (e.g., 2) become smaller numbers (e.g., -2).
        # counter ensures FIFO order for items with the same priority.
        heapq.heappush(self.heap, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.heap:
            return None

        _, _, name, data = heapq.heappop(self.heap)
        return (name, data)
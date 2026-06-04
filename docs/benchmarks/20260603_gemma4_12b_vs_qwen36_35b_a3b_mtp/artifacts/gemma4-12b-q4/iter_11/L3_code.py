import heapq

class JobQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Use negative priority because heapq is a min-heap.
        # A higher priority number (e.g., 2) becomes a smaller 
        # negative number (-2), ensuring it is popped first.
        # self.counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self.heap, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.heap:
            return None

        _, _, name, data = heapq.heappop(self.heap)
        return (name, data)
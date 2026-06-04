import heapq

class JobQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # We use -priority because heapq is a min-heap, but we want higher numbers first.
        # self.counter ensures FIFO order for items with the same priority.
        heapq.heappush(self.heap, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.heap:
            return None
        
        priority_neg, count, name, data = heapq.heappop(self.heap)
        return (name, data)
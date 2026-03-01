import heapq

class JobQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def add_job(self, name: str, details: dict, priority: int = 0):
        heapq.heappush(self.heap, (-priority, self.counter, (name, details)))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.heap:
            return None
        _, _, job = heapq.heappop(self.heap)
        return job
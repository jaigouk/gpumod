import heapq

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # priority: 0=Normal, 1=High, 2=Critical
            # heapq is a min-heap, so we use -priority to make higher numbers come first
            heapq.heappush(self.heap, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.heap:
                return None
            priority_score, count, name, data = heapq.heappop(self.heap)
            return (name, data)
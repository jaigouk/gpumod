import heapq

    class JobQueue:
        def __init__(self):
            self.jobs = []
            self.counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # We use -priority because heapq is a min-heap.
            # Lower number comes out first. -2 is smaller than -1.
            # counter ensures FIFO for same priority.
            heapq.heappush(self.jobs, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.jobs:
                return None
            _, _, name, data = heapq.heappop(self.jobs)
            return (name, data)
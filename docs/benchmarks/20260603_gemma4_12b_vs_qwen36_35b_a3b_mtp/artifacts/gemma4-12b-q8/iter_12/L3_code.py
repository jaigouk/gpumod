import heapq

    class JobQueue:
        def __init__(self):
            self.jobs = []
            self.counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Use -priority to simulate max-heap behavior using min-heap
            # Use self.counter to maintain FIFO for equal priorities
            heapq.heappush(self.jobs, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.jobs:
                return None
            priority_neg, counter, name, data = heapq.heappop(self.jobs)
            return (name, data)
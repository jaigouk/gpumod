import heapq

class JobQueue:
    def __init__(self):
        self.jobs = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # heapq is a min-heap. To prioritize higher numbers, we store negative priority.
        # The counter ensures FIFO behavior for jobs with the same priority level.
        heapq.heappush(self.jobs, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None

        # Pop the item with the lowest value (the highest priority/lowest negative number)
        _, _, name, data = heapq.heappop(self.jobs)
        return (name, data)
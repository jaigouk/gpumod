import heapq

class JobQueue:
    def __init__(self):
        self.jobs = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Negate priority to transform min-heap into max-priority behavior
        # self.counter ensures FIFO order for jobs with the same priority
        heapq.heappush(self.jobs, (-priority, self.counter, name, data))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None

        # Pop the highest priority job (smallest negative value)
        _, _, name, data = heapq.heappop(self.jobs)
        return (name, data)
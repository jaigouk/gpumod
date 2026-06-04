import heapq

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = 0

        def add_job(self, job_name: str, job_data: dict, priority: int = 0):
            # We use -priority because heapq is a min-heap,
            # and we want higher priority numbers to come out first.
            # self.counter ensures FIFO for equal priorities.
            heapq.heappush(self.queue, (-priority, self.counter, job_name, job_data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.queue:
                return None
            _, _, job_name, job_data = heapq.heappop(self.queue)
            return (job_name, job_data)
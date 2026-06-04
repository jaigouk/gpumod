import heapq

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = 0

        def add_job(self, job_id: str, data: dict, priority: int = 0):
            # heapq is a min-heap. Use -priority to simulate a max-heap for priorities.
            # Use self.counter to maintain FIFO order for equal priorities.
            heapq.heappush(self.queue, (-priority, self.counter, job_id, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.queue:
                return None
            priority_score, count, job_id, data = heapq.heappop(self.queue)
            return (job_id, data)
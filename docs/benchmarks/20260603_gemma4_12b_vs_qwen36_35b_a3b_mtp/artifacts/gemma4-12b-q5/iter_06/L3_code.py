import heapq
    from typing import tuple, dict, Optional

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.counter = 0

        def add_job(self, job_id: str, job_data: dict, priority: int = 0):
            # heap is min-heap. We want highest priority (e.g. 2) first.
            # So we store negative priority.
            # Tie-breaker is the counter to ensure FIFO for same priority.
            heapq.heappush(self.heap, (-priority, self.counter, job_id, job_data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.heap:
                return None
            priority_score, count, job_id, job_data = heapq.heappop(self.heap)
            return (job_id, job_data)
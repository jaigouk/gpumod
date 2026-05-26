import heapq
    from typing import Optional, Tuple, Dict

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.counter = 0

        def add_job(self, job_name: str, job_data: dict, priority: int = 0) -> None:
            heapq.heappush(self.heap, (-priority, self.counter, job_name, job_data))
            self.counter += 1

        def get_next_job(self) -> Optional[Tuple[str, dict]]:
            if not self.heap:
                return None
            _, _, job_name, job_data = heapq.heappop(self.heap)
            return (job_name, job_data)
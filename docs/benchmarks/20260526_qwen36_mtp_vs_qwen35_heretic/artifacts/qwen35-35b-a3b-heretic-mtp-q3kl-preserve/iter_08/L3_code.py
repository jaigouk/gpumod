import heapq
    from typing import Optional, Tuple, Dict

    class JobQueue:
        def __init__(self):
            self._heap = []
            self._counter = 0

        def add_job(self, job_name: str, job_data: dict, priority: int = 0) -> None:
            # Negate priority because heapq is a min-heap
            # Higher priority (e.g., 2) should come before lower (e.g., 0)
            # -2 < -1 < 0, so -2 pops first.
            item = (-priority, self._counter, job_name, job_data)
            self._counter += 1
            heapq.heappush(self._heap, item)

        def get_next_job(self) -> Tuple[str, dict] | None:
            if not self._heap:
                return None
            _, _, job_name, job_data = heapq.heappop(self._heap)
            return (job_name, job_data)
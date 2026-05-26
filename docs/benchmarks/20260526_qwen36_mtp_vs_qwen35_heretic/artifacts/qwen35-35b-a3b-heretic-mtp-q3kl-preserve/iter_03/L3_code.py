import heapq
    from typing import Optional, Tuple, Dict

    class JobQueue:
        def __init__(self):
            self._heap = []
            self._counter = 0

        def add_job(self, job_id: str, job_data: Dict, priority: int = 0) -> None:
            # Invert priority for min-heap (higher priority value -> smaller negative value)
            # e.g., 2 -> -2, 1 -> -1, 0 -> 0
            # This ensures priority 2 is popped before 1, which is before 0.
            heap_item = (-priority, self._counter, job_id, job_data)
            heapq.heappush(self._heap, heap_item)
            self._counter += 1

        def get_next_job(self) -> Optional[Tuple[str, Dict]]:
            if not self._heap:
                return None
            _, _, job_id, job_data = heapq.heappop(self._heap)
            return (job_id, job_data)
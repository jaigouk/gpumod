import heapq
    from typing import Optional, Tuple, Dict, Any

    class JobQueue:
        def __init__(self):
            self._heap = []
            self._counter = 0  # To maintain FIFO for same priority

        def add_job(self, name: str, job_data: dict, priority: int = 0) -> None:
            # Priority 2 (Critical) > Priority 1 (High) > Priority 0 (Normal)
            # Min-heap pops smallest. So we negate priority to make larger priority pop first.
            # -2 < -0.
            entry = (-priority, self._counter, name, job_data)
            heapq.heappush(self._heap, entry)
            self._counter += 1

        def get_next_job(self) -> Optional[Tuple[str, dict]]:
            if not self._heap:
                return None
            priority, counter, name, job_data = heapq.heappop(self._heap)
            return (name, job_data)
import heapq

    class JobQueue:
        def __init__(self):
            self._heap = []
            self._sequence = 0

        def add_job(self, job_id: str, job_data: dict, priority: int = 0) -> None:
            # Negate priority so higher priority (e.g., 2) becomes smaller number (-2)
            # to be popped first by min-heap
            entry = (-priority, self._sequence, job_id, job_data)
            heapq.heappush(self._heap, entry)
            self._sequence += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._heap:
                return None
            _, _, job_id, job_data = heapq.heappop(self._heap)
            return (job_id, job_data)
import heapq

class JobQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[int, int, str, dict]] = []
        self._counter = 0

    def add_job(self, job_id: str, payload: dict, priority: int = 0) -> None:
        heapq.heappush(self._heap, (-priority, self._counter, job_id, payload))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job_id, payload = heapq.heappop(self._heap)
        return job_id, payload
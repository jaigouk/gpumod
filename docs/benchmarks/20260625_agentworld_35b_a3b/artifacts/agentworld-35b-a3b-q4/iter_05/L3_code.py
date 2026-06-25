import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, job_id: str, job_data: dict, priority: int = 0):
        entry = [-priority, self._counter, job_id, job_data]
        self._counter += 1
        heapq.heappush(self._heap, entry)

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job_id, job_data = heapq.heappop(self._heap)
        return (job_id, job_data)
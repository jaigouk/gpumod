from collections import deque

class JobQueue:
    def __init__(self) -> None:
        self._queue: deque = deque()
        self._results: dict[str, dict | None] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        self._results[job_id] = None
        return job_id

    def process(self) -> dict | None:
        if not self._queue:
            return None
        job_id, data = self._queue.popleft()
        result = {"status": "completed", "data": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)
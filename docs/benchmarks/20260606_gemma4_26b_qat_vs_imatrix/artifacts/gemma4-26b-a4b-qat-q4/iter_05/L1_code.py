from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self, worker_func) -> None:
        """
        Processes the next job in the queue using the provided worker function.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            result = worker_func(data)
            self._results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)
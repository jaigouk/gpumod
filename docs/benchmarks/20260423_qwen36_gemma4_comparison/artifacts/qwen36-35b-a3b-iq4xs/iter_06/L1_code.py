import collections

class JobQueue:
    def __init__(self):
        self._queue = collections.deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append(job_id)
        self._results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)
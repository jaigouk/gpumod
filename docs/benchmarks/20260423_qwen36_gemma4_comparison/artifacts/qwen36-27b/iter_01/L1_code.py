import collections

class JobQueue:
    def __init__(self):
        self._queue = collections.deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append({"job_id": job_id, "data": data})
        return job_id

    def process_next(self) -> str | None:
        if not self._queue:
            return None
        job = self._queue.popleft()
        job_id = job["job_id"]
        self._results[job_id] = {"status": "completed", "input": job["data"]}
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)
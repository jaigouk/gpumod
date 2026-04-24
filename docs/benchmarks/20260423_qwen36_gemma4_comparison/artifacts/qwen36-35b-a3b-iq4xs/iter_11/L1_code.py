from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append({"id": job_id, "data": data})
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id, None)

    def process(self) -> dict | None:
        if self._queue:
            job = self._queue.popleft()
            result = {"status": "completed", "data": job["data"]}
            self._results[job["id"]] = result
            return result
        return None

    def process_all(self):
        while self._queue:
            self.process()
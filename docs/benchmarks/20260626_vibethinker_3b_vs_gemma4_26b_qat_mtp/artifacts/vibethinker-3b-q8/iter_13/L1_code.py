from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._data = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._data:
            raise ValueError(f"Job {job_id} already exists")
        self._queue.append(job_id)
        self._data[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        if job_id not in self._data:
            return None
        # Process any jobs that were added before the requested one
        while self._queue and self._queue[0] != job_id:
            front_id = self._queue.popleft()
            del self._data[front_id]
        # Now the requested job should be at the front
        if not self._queue or self._queue[0] != job_id:
            return None
        self._queue.popleft()
        return self._data.pop(job_id)
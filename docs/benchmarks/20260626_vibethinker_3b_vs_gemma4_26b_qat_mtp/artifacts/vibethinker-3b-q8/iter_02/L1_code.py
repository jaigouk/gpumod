from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        # Ensure unique job_id
        if job_id in self._results or job_id in self._queue:
            raise ValueError(f"Job {job_id} already exists.")
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        # Return immediately if already processed
        if job_id in self._results:
            return self._results[job_id]

        # Process jobs in FIFO order until the requested one is completed
        while self._queue:
            pid, data = self._queue.popleft()
            # Simple processing: result is the original data (could be replaced)
            result = data
            self._results[pid] = result
            if pid == job_id:
                return result
        # Job not found
        return None
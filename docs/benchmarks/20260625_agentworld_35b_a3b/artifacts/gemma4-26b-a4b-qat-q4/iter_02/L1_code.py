from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue in FIFO order."""
        self._queue.append((job_id, data))
        return job_id

    def process_next(self) -> None:
        """Processes the next job in the queue."""
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulating processing by treating the input data as the result
            self._results[job_id] = data

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not found."""
        return self._results.get(job_id)
from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        self._jobs = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next_job(self, result: dict):
        """Processes the next job in FIFO order and stores its result."""
        if self._queue:
            job_id = self._queue.popleft()
            self._results[job_id] = result
            self._jobs.pop(job_id, None)

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)
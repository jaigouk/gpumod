from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self, worker_func) -> str | None:
        """
        Processes the next job in the FIFO queue using the provided worker function.
        The worker function should accept the job data and return a dictionary.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()
        result = worker_func(data)
        self._results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Gets the result of a completed job."""
        return self._results.get(job_id)
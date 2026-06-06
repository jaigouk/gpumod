from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next(self, task_func):
        """
        Processes the next job in the queue in FIFO order.
        task_func should accept the job data and return a result dict.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()
        result = task_func(data)
        self._results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Get the result of a completed job."""
        return self._results.get(job_id)
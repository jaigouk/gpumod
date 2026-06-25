from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def process_next_job(self):
        """Processes the next job in the queue in FIFO order."""
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._jobs.pop(job_id)

        # Simulate processing the task
        result = {"status": "success", "original_data": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job or None if not found."""
        return self._results.get(job_id)
from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next(self) -> dict | None:
        """Processes the next job in the queue in FIFO order."""
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._jobs.pop(job_id)

        # Simulate processing: returns the input data wrapped in a result dict
        result = {"status": "completed", "data": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)
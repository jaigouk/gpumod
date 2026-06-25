from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        self._pending_jobs = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._pending_jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next_job(self):
        """Processes the next job in the queue in FIFO order."""
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._pending_jobs.pop(job_id)

        # Simulate processing logic
        result = {"input_data": data, "status": "completed"}
        self._results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a job if it has been processed."""
        return self._results.get(job_id)
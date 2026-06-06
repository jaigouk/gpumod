from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._pending_jobs = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._queue.append(job_id)
        self._pending_jobs[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)

    def process_next_job(self):
        """
        Processes the next job in the queue in FIFO order.
        In a real-world scenario, this would be called by a worker process.
        """
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._pending_jobs.pop(job_id)

        # Simulate processing logic
        result = {"status": "completed", "processed_data": data}
        self._results[job_id] = result
        return result
from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> dict | None:
        """
        Processes the next job in the queue using FIFO order.
        Returns the result of the processed job.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()

        # Simulate processing logic
        result = {"status": "completed", "data": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)
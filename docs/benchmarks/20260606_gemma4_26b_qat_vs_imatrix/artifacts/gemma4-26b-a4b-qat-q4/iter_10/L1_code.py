from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> str | None:
        """
        Processes the next job in the queue in FIFO order.
        This method is necessary to fulfill the 'processed' requirement.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()
        # Simulate processing by wrapping the data in a result dictionary
        self._results[job_id] = {"status": "success", "data": data}
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed or not found."""
        return self._results.get(job_id)
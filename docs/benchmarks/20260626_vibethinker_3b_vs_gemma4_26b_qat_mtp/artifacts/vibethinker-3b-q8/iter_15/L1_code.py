from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a new job to the end of the queue and return its ID."""
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Return the result of a completed job. processes jobs in FIFO order."""
        # If the result is already computed, return it directly.
        if job_id in self._results:
            return self._results[job_id]

        # Process jobs from the queue until the requested job is completed.
        while self._queue:
            current_job_id, current_data = self._queue.popleft()
            # Simulated processing: the result is the original data (could be transformed).
            result = current_data
            self._results[current_job_id] = result
            if current_job_id == job_id:
                break

        return self._results.get(job_id)
from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self):
        """
        Simulates the processing of the next job in FIFO order.
        In a real scenario, this might be handled by a worker thread.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()

        # Simulated processing: returning the input data as the result
        self._results[job_id] = {
            "status": "completed",
            "data": data
        }
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job or None if not finished."""
        return self._results.get(job_id)
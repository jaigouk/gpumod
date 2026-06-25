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
        Processes the next job in the queue (FIFO).
        In a real-world scenario, this might run in a separate thread or worker.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()

        # Simulate processing logic
        result = {"status": "completed", "output": data}

        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> dict | None:
        """Gets the result of a completed job."""
        return self._results.get(job_id)
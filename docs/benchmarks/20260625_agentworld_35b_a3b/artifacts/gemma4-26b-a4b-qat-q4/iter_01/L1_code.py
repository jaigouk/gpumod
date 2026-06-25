from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)

    def process_next_job(self):
        """Processes the next job in the queue in FIFO order."""
        if not self._queue:
            return

        job_id, data = self._queue.popleft()

        # Simulate processing logic (returning the data as the result)
        self._results[job_id] = {"status": "completed", "output": data}
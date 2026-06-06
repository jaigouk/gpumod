from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next_job(self) -> str | None:
        """
        Processes the next job in FIFO order.
        In a real-world scenario, this might be called by a worker thread.
        """
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._jobs.pop(job_id)

        # Simulate job processing logic
        result = {"processed_data": data, "status": "completed"}
        self._results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)
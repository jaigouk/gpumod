from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next_job(self) -> str | None:
        """
        Processes the next job in the queue following FIFO order.
        """
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        job_data = self._jobs.pop(job_id, None)

        if job_data is not None:
            # Simulate processing by returning a result dictionary
            self._results[job_id] = {"status": "completed", "output": job_data}
            return job_id
        return None

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)